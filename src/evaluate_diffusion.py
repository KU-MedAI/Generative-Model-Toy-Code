import argparse
import os
import sys
import multiprocessing
sys.setrecursionlimit(10000)

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    if not all_ring_sizes:
        logger.info('ring ratio: None (no rings evaluated)')
        return
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def dock_worker(mol, ligand_filename, protein_root, docking_mode, exhaustiveness, task_idx, return_dict):
    try:
        if docking_mode == 'qvina':
            vina_task = QVinaDockingTask.from_generated_mol(
                mol, ligand_filename, protein_root=protein_root)
            vina_results = vina_task.run_sync()
            return_dict[task_idx] = vina_results
        elif docking_mode in ['vina_score', 'vina_dock']:
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename, protein_root=protein_root)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
            vina_results = {
                'score_only': score_only_results,
                'minimize': minimize_results
            }
            if docking_mode == 'vina_dock':
                docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
                vina_results['dock'] = docking_results
            return_dict[task_idx] = vina_results
        else:
            return_dict[task_idx] = None
    except Exception as e:
        return_dict[f'error_{task_idx}'] = str(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=20, help='중간 저장 주기 (분자 개수)')
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    checkpoint_path = os.path.join(result_path, f'checkpoint_step_{args.eval_step}.pt')
    
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # =========================================================================
    # [수정된 부분] 특정 범위(data_0 ~ data_20) 탐색 및 빈 sdf 폴더 스킵 로직
    # =========================================================================
    results_fn_list = []
    for i in range(1):  # 0부터 20까지 (data_0 ~ data_20)
        data_dir = os.path.join(args.sample_path, f'data_{i}')
        sdf_dir = os.path.join(data_dir, 'sdf')
        
        # 1. 폴더가 아예 존재하지 않으면 패스
        if not os.path.isdir(data_dir):
            continue
            
        # 2. sdf 폴더가 없거나 내부 파일이 없는 빈 폴더면 건너뜀
        if not os.path.isdir(sdf_dir) or len(os.listdir(sdf_dir)) == 0:
            logger.info(f"건너뜀: {data_dir} (sdf 폴더가 없거나 비어있음)")
            continue
            
        # 3. 해당 폴더 안에서 .pt 파일(결과 파일) 가져오기
        pt_files = glob(os.path.join(data_dir, '*result*.pt'))
        if not pt_files:  # result.pt를 못 찾으면 result 파일명 무관하게 pt 탐색
            pt_files = glob(os.path.join(data_dir, '*.pt'))
            
        if not pt_files:
            logger.info(f"건너뜀: {data_dir} (평가할 .pt 파일이 없음)")
            continue
            
        results_fn_list.extend(pt_files)

    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    
    num_examples = len(results_fn_list)
    logger.info(f'데이터 로드 완료: 필터링 후 총 {num_examples}개 파일 (data_0 ~ data_20). {args.num_workers}개 CPU 사용.')
    # =========================================================================

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    
    valid_tasks = []

    # 1단계: RDKit 기반 빠른 검사
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='1. Fast Check')):
        r = torch.load(r_name)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)

        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except (reconstruct.MolReconsError, RecursionError):
                continue
            
            n_recon_success += 1
            if '.' in smiles: continue
            n_complete += 1

            try:
                chem_results = scoring_func.get_chem(mol)
                valid_tasks.append({
                    'example_idx': example_idx, 'sample_idx': sample_idx,
                    'mol': mol, 'smiles': smiles, 'pred_pos': pred_pos, 'pred_v': pred_v,
                    'chem_results': chem_results, 'ligand_filename': r['data'].ligand_filename,
                    'pair_dist': pair_dist, 'pred_atom_type': pred_atom_type
                })
            except Exception: continue

    # 2단계: 병렬 도킹 및 중간 저장
    logger.info(f"병렬 도킹 시작: {len(valid_tasks)}개 분자 대상")
    manager = multiprocessing.Manager()
    
    for i in tqdm(range(0, len(valid_tasks), args.num_workers), desc='2. Parallel Docking'):
        chunk = valid_tasks[i:i+args.num_workers]
        return_dict = manager.dict()
        processes = []

        for j, task in enumerate(chunk):
            p = multiprocessing.Process(target=dock_worker, args=(
                task['mol'], task['ligand_filename'], args.protein_root, 
                args.docking_mode, args.exhaustiveness, j, return_dict
            ))
            processes.append(p)
            p.start()

        for p in processes: p.join()
            
        for j, task in enumerate(chunk):
            p = processes[j]
            if p.exitcode != 0: continue
            if f'error_{j}' in return_dict: continue
            
            n_eval_success += 1
            bond_dist = eval_bond_length.bond_distance_from_mol(task['mol'])
            all_bond_dist += bond_dist
            success_pair_dist += task['pair_dist']
            success_atom_types += Counter(task['pred_atom_type'])

            results.append({
                'mol': task['mol'], 'smiles': task['smiles'],
                'ligand_filename': task['ligand_filename'], 'pred_pos': task['pred_pos'],
                'pred_v': task['pred_v'], 'chem_results': task['chem_results'],
                'vina': return_dict.get(j, None)
            })

        # 중간 저장
        if len(results) > 0 and len(results) % args.save_interval == 0:
            logger.info(f"--- 중간 저장 중... (현재 완료: {len(results)}개) ---")
            torch.save({
                'all_results': results,
                'n_eval_success': n_eval_success,
                'num_samples_so_far': num_samples
            }, checkpoint_path)

    logger.info(f'평가 완료! 총 {num_samples}개 샘플.')
    
    fraction_mol_stable = all_mol_stable / num_samples if num_samples > 0 else 0.0
    fraction_atm_stable = all_atom_stable / all_n_atom if all_n_atom > 0 else 0.0
    fraction_recon = n_recon_success / num_samples if num_samples > 0 else 0.0
    fraction_eval = n_eval_success / num_samples if num_samples > 0 else 0.0
    fraction_complete = n_complete / num_samples if num_samples > 0 else 0.0
    
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

    if all_bond_dist:
        c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
        c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
        logger.info('JS bond distances of complete mols: ')
        print_dict(c_bond_length_dict, logger)

    if success_pair_dist:
        success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
        success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
        print_dict(success_js_metrics, logger)
        
        if args.save:
            eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                                metrics=success_js_metrics,
                                                save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    if len(success_atom_types) > 0:
        atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
        logger.info('Atom type JS: %.4f' % atom_type_js)
    else:
        logger.info('Atom type JS: None (No successful atoms parsed)')

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    if results:
        qed = [r['chem_results']['qed'] for r in results]
        sa = [r['chem_results']['sa'] for r in results]
        logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
        logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
        
        if args.docking_mode == 'qvina':
            vina = [r['vina'][0]['affinity'] for r in results]
            logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
        elif args.docking_mode in ['vina_dock', 'vina_score']:
            vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
            vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
            logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
            logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
            if args.docking_mode == 'vina_dock':
                vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
                logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

        print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results
        }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))