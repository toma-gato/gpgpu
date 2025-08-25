import sqlite3
import json
import argparse
import os
import re
import glob
from collections import defaultdict
from datetime import datetime

def clean_operation_name(name):
    """Nettoie le nom de l'opération en retirant le suffixe de version et filtre cudaDeviceSynchronize"""
    if name is None:
        return None
    name = re.sub(r'_v\d+$', '', name)
    return None if name == "cudaDeviceSynchronize" else name

def extract_cuda_summary(db_path):
    """Extrait les données détaillées pour les tableaux LaTeX"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        results = defaultdict(lambda: {'time': 0.0, 'instances': 0, 'category': ''})

        # 1. Kernel executions
        cursor.execute("""
            SELECT s.value, COUNT(*), SUM(k.end - k.start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON k.shortName = s.id
            GROUP BY s.value
        """)
        for kernel, count, total_time in cursor.fetchall():
            kernel_clean = clean_operation_name(kernel)
            if kernel_clean is not None:
                results[kernel_clean] = {
                    'time': total_time,
                    'instances': count,
                    'category': 'CUDA_KERNEL'
                }

        # 2. Runtime APIs
        cursor.execute("""
            SELECT s.value, COUNT(*), SUM(r.end - r.start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON r.nameId = s.id
            GROUP BY s.value
        """)
        for api_name, count, total_time in cursor.fetchall():
            clean_name = api_name.split('(')[0].strip()
            clean_name = clean_operation_name(clean_name)
            if clean_name is not None:
                results[clean_name] = {
                    'time': total_time,
                    'instances': count,
                    'category': 'CUDA_API'
                }

        # 3. Memory operations
        cursor.execute("""
            SELECT 
                CASE copyKind
                    WHEN 1 THEN '[CUDA memcpy Host-to-Device]'
                    WHEN 2 THEN '[CUDA memcpy Device-to-Host]'
                END AS operation,
                COUNT(*),
                SUM(end - start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind IN (1, 2)
            GROUP BY operation
        """)
        for operation, count, total_time in cursor.fetchall():
            results[operation] = {
                'time': total_time,
                'instances': count,
                'category': 'MEMORY_OPER'
            }

        return dict(results)
    
    except sqlite3.Error as e:
        print(f"Erreur SQLite: {e}")
        return None
    finally:
        if conn:
            conn.close()

def extract_memops_summary(db_path):
    """Extrait les statistiques de transfert mémoire pour les tableaux LaTeX"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        results = {}

        cursor.execute("""
            SELECT 
                CASE copyKind
                    WHEN 1 THEN 'Host-to-Device'
                    WHEN 2 THEN 'Device-to-Host'
                END AS direction,
                COUNT(*),
                SUM(bytes) AS total_bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind IN (1, 2)
            GROUP BY direction
        """)

        for direction, count, total_bytes in cursor.fetchall():
            results[direction] = {
                'bytes': total_bytes / (1024 * 1024),
                'count': count
            }

        return results
    
    except sqlite3.Error as e:
        print(f"Erreur SQLite: {e}")
        return None
    finally:
        if conn:
            conn.close()

def save_to_json(version, cuda_data, mem_data, output_dir):
    """Ajoute les données aux fichiers JSON avec un timestamp"""
    timestamp = datetime.now().isoformat()
    new_entry = {
        "version": version,
        "timestamp": timestamp,
        "cuda_data": cuda_data,
        "mem_data": mem_data
    }
    
    # Chemins des fichiers JSON
    cuda_path = os.path.join(output_dir, "cuda_summary_latex.json")
    mem_path = os.path.join(output_dir, "memops_summary_latex.json")
    
    # CUDA: Charger, mettre à jour, sauvegarder
    cuda_entries = []
    if os.path.exists(cuda_path):
        try:
            with open(cuda_path, 'r') as f:
                cuda_entries = json.load(f)
        except json.JSONDecodeError:
            cuda_entries = []
    
    cuda_entries.append(new_entry)
    
    with open(cuda_path, 'w') as f:
        json.dump(cuda_entries, f, indent=4)

    # Mémoire: Charger, mettre à jour, sauvegarder
    mem_entries = []
    if os.path.exists(mem_path):
        try:
            with open(mem_path, 'r') as f:
                mem_entries = json.load(f)
        except json.JSONDecodeError:
            mem_entries = []
    
    mem_entries.append(new_entry)
    
    with open(mem_path, 'w') as f:
        json.dump(mem_entries, f, indent=4)

    return True

def main():
    parser = argparse.ArgumentParser(description='Extract detailed CUDA profiling data for LaTeX tables')
    parser.add_argument('input_path', help='Chemin vers un fichier.sqlite ou dossier contenant des .sqlite')
    parser.add_argument('--version', required=True, help='Numéro de version pour les données')
    parser.add_argument('--output', help='Dossier de sortie pour les fichiers JSON', default='.')
    args = parser.parse_args()

    # Créer le dossier de sortie
    os.makedirs(args.output, exist_ok=True)
    
    # Liste des fichiers à traiter
    if os.path.isfile(args.input_path) and args.input_path.endswith('.sqlite'):
        files = [args.input_path]
    elif os.path.isdir(args.input_path):
        files = glob.glob(os.path.join(args.input_path, '*.sqlite'))
    else:
        print("Le chemin doit être un fichier .sqlite ou un dossier contenant des .sqlite")
        return

    # Traiter chaque fichier avec la même version
    version = args.version
    total_entries = 0
    
    for sqlite_path in files:
        source_file = os.path.basename(sqlite_path)
        
        print(f"\nTraitement de {source_file} [version: {version}]")
        
        cuda_data = extract_cuda_summary(sqlite_path)
        mem_data = extract_memops_summary(sqlite_path)
        
        if cuda_data is None or mem_data is None:
            print(f"Échec du traitement pour {source_file}")
            continue
        
        if save_to_json(version, cuda_data, mem_data, args.output):
            total_entries += 1
            print("Succès!")
        else:
            print("Échec de sauvegarde")

    print(f"\nTotal: {total_entries} fichier(s) traités avec succès pour la version '{version}'")

if __name__ == "__main__":
    main()
