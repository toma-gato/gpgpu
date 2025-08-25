import sqlite3
import json
import argparse
import os
import re
import glob
from collections import defaultdict
from datetime import datetime

def clean_operation_name(name):
    """Nettoie le nom de l'opération en retirant le suffixe de version"""
    if name is None:
        return None
    name = re.sub(r'_v\d+$', '', name)
    return None if name == "cuModuleGetLoadingMode" else name

def extract_cuda_summary(db_path):
    """Extrait les données de temps d'exécution CUDA depuis la base SQLite"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        results = defaultdict(float)

        # 1. Kernel executions
        cursor.execute("""
            SELECT s.value, SUM(k.end - k.start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON k.shortName = s.id
            GROUP BY s.value
        """)
        for kernel, total_time in cursor.fetchall():
            if clean_name := clean_operation_name(kernel):
                results[clean_name] = total_time

        # 2. Runtime APIs
        cursor.execute("""
            SELECT s.value, SUM(r.end - r.start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON r.nameId = s.id
            GROUP BY s.value
        """)
        for api_name, total_time in cursor.fetchall():
            clean_name = api_name.split('(')[0].strip()
            if clean_name := clean_operation_name(clean_name):
                results[clean_name] += total_time

        # 3. Memory operations
        cursor.execute("""
            SELECT 
                CASE copyKind
                    WHEN 1 THEN '[CUDA memcpy Host-to-Device]'
                    WHEN 2 THEN '[CUDA memcpy Device-to-Host]'
                END,
                SUM(end - start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind IN (1, 2)
            GROUP BY copyKind
        """)
        for operation, total_time in cursor.fetchall():
            results[operation] = total_time

        return dict(results)
    
    except sqlite3.Error as e:
        print(f"Erreur SQLite: {e}")
        return None
    finally:
        if conn:
            conn.close()

def extract_memops_summary(db_path):
    """Extrait les statistiques de transfert mémoire"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        results = {}

        cursor.execute("""
            SELECT 
                CASE copyKind
                    WHEN 1 THEN 'Host-to-Device'
                    WHEN 2 THEN 'Device-to-Host'
                END,
                SUM(bytes) / (1024 * 1024)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind IN (1, 2)
            GROUP BY copyKind
        """)
        
        for direction, total_mb in cursor.fetchall():
            results[direction] = total_mb

        return results
    
    except sqlite3.Error as e:
        print(f"Erreur SQLite: {e}")
        return None
    finally:
        if conn:
            conn.close()

def save_to_json(version, source_file, cuda_data, memops_data, output_dir):
    """Ajoute les données aux fichiers JSON avec un timestamp"""
    timestamp = datetime.now().isoformat()
    
    # Structure pour CUDA
    cuda_entry = {
        "version": version,
        "timestamp": timestamp,
        "source_file": source_file,
        "data": cuda_data
    }
    
    # Structure pour mémoire
    memops_entry = {
        "version": version,
        "timestamp": timestamp,
        "source_file": source_file,
        "data": memops_data
    }

    # Chemin des fichiers JSON
    cuda_json_path = os.path.join(output_dir, "cuda_summary.json")
    memops_json_path = os.path.join(output_dir, "memops_summary.json")

    # CUDA: Charger, mettre à jour, sauvegarder
    cuda_entries = []
    if os.path.exists(cuda_json_path):
        try:
            with open(cuda_json_path, 'r') as f:
                cuda_entries = json.load(f)
        except json.JSONDecodeError:
            cuda_entries = []
    
    cuda_entries.append(cuda_entry)
    
    with open(cuda_json_path, 'w') as f:
        json.dump(cuda_entries, f, indent=4)

    # Mémoire: Charger, mettre à jour, sauvegarder
    memops_entries = []
    if os.path.exists(memops_json_path):
        try:
            with open(memops_json_path, 'r') as f:
                memops_entries = json.load(f)
        except json.JSONDecodeError:
            memops_entries = []
    
    memops_entries.append(memops_entry)
    
    with open(memops_json_path, 'w') as f:
        json.dump(memops_entries, f, indent=4)

    return True

def main():
    parser = argparse.ArgumentParser(description='Extract CUDA profiling data from Nsight Systems SQLite reports')
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
    
    for sqlite_path in files:
        source_file = os.path.basename(sqlite_path)
        
        print(f"\nTraitement de {source_file} [version: {version}]")
        
        cuda_data = extract_cuda_summary(sqlite_path)
        memops_data = extract_memops_summary(sqlite_path)
        
        if cuda_data is None or memops_data is None:
            print(f"Échec du traitement pour {source_file}")
            continue
        
        if save_to_json(version, source_file, cuda_data, memops_data, args.output):
            print("Succès!")
        else:
            print("Échec de sauvegarde")

if __name__ == "__main__":
    main()
