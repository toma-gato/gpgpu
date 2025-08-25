import json
import pandas as pd
import os
import argparse
import numpy as np

def escape_latex(text):
    """Échappe les caractères spéciaux LaTeX"""
    if not isinstance(text, str):
        return text
    return text.replace('_', '\\_')

def aggregate_cuda_operations(cuda_entries):
    """Agrège les données CUDA en calculant moyenne et écart-type"""
    operations = {}
    
    for entry in cuda_entries:
        for op, data in entry.items():
            if op not in operations:
                operations[op] = {
                    'time': [],
                    'instances': [],
                    'category': data['category']
                }
            operations[op]['time'].append(data['time'])
            operations[op]['instances'].append(data['instances'])
    
    aggregated = {}
    for op, values in operations.items():
        time_values = np.array(values['time'])
        instances_values = np.array(values['instances'])
        
        aggregated[op] = {
            'time_mean': np.mean(time_values),
            'time_std': np.std(time_values, ddof=1) if len(time_values) > 1 else 0,
            'instances_mean': np.mean(instances_values),
            'instances_std': np.std(instances_values, ddof=1) if len(instances_values) > 1 else 0,
            'category': values['category']
        }
    
    return aggregated

def aggregate_mem_operations(mem_entries):
    """Agrège les données mémoire en calculant moyenne et écart-type"""
    operations = {}
    
    for entry in mem_entries:
        for op, data in entry.items():
            if op not in operations:
                operations[op] = {
                    'bytes': [],
                    'count': []
                }
            operations[op]['bytes'].append(data['bytes'])
            operations[op]['count'].append(data['count'])
    
    aggregated = {}
    for op, values in operations.items():
        bytes_values = np.array(values['bytes'])
        count_values = np.array(values['count'])
        
        aggregated[op] = {
            'bytes_mean': np.mean(bytes_values),
            'bytes_std': np.std(bytes_values, ddof=1) if len(bytes_values) > 1 else 0,
            'count_mean': np.mean(count_values),
            'count_std': np.std(count_values, ddof=1) if len(count_values) > 1 else 0
        }
    
    return aggregated

def generate_latex_tables(cuda_file="cuda_summary.json", mem_file="memops_summary.json", output_dir="."):
    # Charger les données
    with open(cuda_file, 'r') as f:
        cuda_data = json.load(f)
    with open(mem_file, 'r') as f:
        mem_data = json.load(f)
    
    # Organiser les données par version
    cuda_by_version = {}
    mem_by_version = {}
    
    for entry in cuda_data:
        version = entry["version"]
        if version not in cuda_by_version:
            cuda_by_version[version] = []
        cuda_by_version[version].append(entry["cuda_data"])
    
    for entry in mem_data:
        version = entry["version"]
        if version not in mem_by_version:
            mem_by_version[version] = []
        mem_by_version[version].append(entry["mem_data"])
    
    # Générer les tableaux LaTeX pour chaque version
    for version in cuda_by_version.keys():
        if version not in mem_by_version:
            continue
            
        # Agrégation des données CUDA
        cuda_aggregated = aggregate_cuda_operations(cuda_by_version[version])
        
        # Calcul des totaux CUDA
        total_time = sum(data['time_mean'] for data in cuda_aggregated.values())
        total_instances = sum(data['instances_mean'] for data in cuda_aggregated.values())
        
        # Création du DataFrame CUDA
        cuda_df = pd.DataFrame([
            {
                'operation': op,
                'time_mean': data['time_mean'],
                'time_std': data['time_std'],
                'instances_mean': data['instances_mean'],
                'instances_std': data['instances_std'],
                'category': data['category']
            }
            for op, data in cuda_aggregated.items()
        ])
        
        # Calcul des pourcentages et tri
        cuda_df['percentage'] = (cuda_df['time_mean'] / total_time) * 100
        cuda_df = cuda_df.sort_values('time_mean', ascending=False)
        
        # Agrégation des données mémoire
        mem_aggregated = aggregate_mem_operations(mem_by_version[version])
        
        # Calcul des totaux mémoire
        total_bytes = sum(data['bytes_mean'] for data in mem_aggregated.values())
        total_count = sum(data['count_mean'] for data in mem_aggregated.values())
        
        # Création du DataFrame mémoire
        mem_df = pd.DataFrame([
            {
                'operation': op,
                'bytes_mean': data['bytes_mean'],
                'bytes_std': data['bytes_std'],
                'count_mean': data['count_mean'],
                'count_std': data['count_std']
            }
            for op, data in mem_aggregated.items()
        ])
        
        # Générer le tableau LaTeX pour CUDA
        latex_cuda = f"""\\begin{{table}}[h]
\\centering
\\caption{{Analyse des temps d'exécution CUDA - {version}}}
\\label{{tab:cuda_timing_{version.replace(" ", "_")}}}
\\begin{{tabular}}{{|r|r|r|r|r|l|l|}}
\\hline
\\textbf{{\\% Time}} & \\textbf{{Mean Time (ms)}} & \\textbf{{σ Time}} & \\textbf{{Mean Instances}} & \\textbf{{σ Instances}} & \\textbf{{Category}} & \\textbf{{Operation}} \\\\
\\hline
"""
        
        for _, row in cuda_df.iterrows():
            escaped_category = escape_latex(row['category'])
            escaped_operation = escape_latex(row['operation'])
            
            latex_cuda += (
                f"{row['percentage']:.1f} \\% & "
                f"{row['time_mean']:,.3f} & "
                f"{row['time_std']:,.3f} & "
                f"{row['instances_mean']:,.1f} & "
                f"{row['instances_std']:,.1f} & "
                f"{escaped_category} & "
                f"{escaped_operation} \\\\\n\\hline\n"
            )
        
        # Ajouter la ligne de total
        latex_cuda += f" & \\textbf{{{total_time:,.3f} ms}} & & \\textbf{{{total_instances:,.1f}}} & & & \\\\\n\\hline\n"
        latex_cuda += "\\end{tabular}\n\\end{table}\n"
        
        # Générer le tableau LaTeX pour mémoire
        latex_mem = f"""\\begin{{table}}[h]
\\centering
\\caption{{Analyse des opérations mémoire - {version}}}
\\label{{tab:memory_ops_{version.replace(" ", "_")}}}
\\begin{{tabular}}{{|r|r|r|r|l|}}
\\hline
\\textbf{{Mean Bytes (MiB)}} & \\textbf{{σ Bytes}} & \\textbf{{Mean Count}} & \\textbf{{σ Count}} & \\textbf{{Opération}} \\\\
\\hline
"""
        
        for _, row in mem_df.iterrows():
            escaped_operation = escape_latex(row['operation'])
            latex_mem += (
                f"{row['bytes_mean']:,.2f} & "
                f"{row['bytes_std']:,.2f} & "
                f"{row['count_mean']:,.1f} & "
                f"{row['count_std']:,.1f} & "
                f"{escaped_operation} \\\\\n\\hline\n"
            )
        
        # Ajouter la ligne de total
        latex_mem += f"\\textbf{{{total_bytes:,.2f} MiB}} & & \\textbf{{{total_count:,.1f}}} & & \\\\\n\\hline\n"
        latex_mem += "\\end{tabular}\n\\end{table}\n"
        
        # Sauvegarder dans un fichier
        output_path = os.path.join(output_dir, f"latex_tables_{version.replace(' ', '_')}.txt")
        with open(output_path, 'w') as f:
            f.write("% ===== TABLEAU CUDA =====\n")
            f.write(latex_cuda)
            f.write("\n\n% ===== TABLEAU MÉMOIRE =====\n")
            f.write(latex_mem)
        
        print(f"Tableaux LaTeX pour {version} sauvegardés dans {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Génère des tableaux LaTeX à partir des données de profiling')
    parser.add_argument('--cuda_file', default="cuda_summary_latex.json", help='Fichier JSON des données CUDA')
    parser.add_argument('--mem_file', default="memops_summary_latex.json", help='Fichier JSON des données mémoire')
    parser.add_argument('--output_dir', default=".", help='Dossier de sortie pour les fichiers texte')
    args = parser.parse_args()
    
    generate_latex_tables(args.cuda_file, args.mem_file, args.output_dir)
