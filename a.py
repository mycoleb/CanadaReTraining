"""
Analyze cluster exemplars to understand which programs fall into each cluster.
"""
from pathlib import Path
import pandas as pd
import sys


def parse_program_key(program_key: str) -> dict:
    """
    Parse the program_key string into its components.
    
    Example: "2019 | Canada | Bachelor's degree | Engineering"
    Returns: {'year': '2019', 'geo': 'Canada', 'level': "Bachelor's degree", 'field': 'Engineering'}
    """
    parts = [p.strip() for p in program_key.split('|')]
    
    # The structure depends on what dimensions were included
    # Typically: REF_DATE | GEO | Level of study | Field of study | ...
    result = {}
    
    if len(parts) >= 1:
        result['year'] = parts[0]
    if len(parts) >= 2:
        result['geo'] = parts[1]
    if len(parts) >= 3:
        result['level'] = parts[2]
    if len(parts) >= 4:
        result['field'] = parts[3]
    if len(parts) >= 5:
        result['sex'] = parts[4]
    if len(parts) >= 6:
        result['additional'] = ' | '.join(parts[5:])
    
    return result


def main():
    # Try multiple possible locations
    possible_paths = [
        Path("outputs/retraining_cluster_exemplars.csv"),
        Path("./outputs/retraining_cluster_exemplars.csv"),
        Path("../outputs/retraining_cluster_exemplars.csv"),
        Path.cwd() / "outputs" / "retraining_cluster_exemplars.csv",
    ]
    
    exemplars_path = None
    for path in possible_paths:
        if path.exists():
            exemplars_path = path
            break
    
    if exemplars_path is None:
        print("ERROR: Could not find retraining_cluster_exemplars.csv")
        print("\nSearched in:")
        for path in possible_paths:
            print(f"  - {path.resolve()}")
        print(f"\nCurrent directory: {Path.cwd()}")
        print("\nPlease run from the project root directory (CanadianReTraining)")
        return 1
    
    print("=" * 80)
    print("CLUSTER ANALYSIS: Which Programs Are in Each Cluster?")
    print("=" * 80)
    print(f"\nReading from: {exemplars_path.resolve()}")
    
    df = pd.read_csv(exemplars_path)
    
    print(f"Loaded {len(df)} exemplar programs")
    print(f"Columns: {list(df.columns)}\n")
    
    # Parse the program keys
    parsed = df['program_key'].apply(parse_program_key)
    parsed_df = pd.DataFrame(parsed.tolist())
    
    # Combine with original data
    analysis = pd.concat([df[['cluster', 'program_key', 'distance_to_center']], parsed_df], axis=1)
    
    # Get unique clusters
    clusters = sorted(df['cluster'].unique())
    
    print("=" * 80)
    print("DETAILED BREAKDOWN BY CLUSTER")
    print("=" * 80)
    
    for cluster_id in clusters:
        cluster_data = analysis[analysis['cluster'] == cluster_id].copy()
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id}: {len(cluster_data)} exemplar programs")
        print(f"{'='*80}")
        
        # Show closest examples to cluster center
        cluster_data = cluster_data.sort_values('distance_to_center')
        
        print("\nTop 10 most representative programs (closest to cluster center):")
        print("-" * 80)
        
        for i, (idx, row) in enumerate(cluster_data.head(10).iterrows(), 1):
            print(f"\n{i}. Distance to center: {row['distance_to_center']:.3f}")
            if 'year' in row and pd.notna(row['year']):
                print(f"   Year: {row['year']}")
            if 'geo' in row and pd.notna(row['geo']):
                print(f"   Geography: {row['geo']}")
            if 'level' in row and pd.notna(row['level']):
                print(f"   Level: {row['level']}")
            if 'field' in row and pd.notna(row['field']):
                print(f"   Field: {row['field']}")
            if 'sex' in row and pd.notna(row['sex']):
                print(f"   Sex: {row['sex']}")
            if 'additional' in row and pd.notna(row['additional']):
                print(f"   Additional: {row['additional']}")
        
        # Summary statistics
        print(f"\n{'-'*80}")
        print("CLUSTER SUMMARY:")
        print(f"{'-'*80}")
        
        if 'level' in cluster_data.columns:
            print(f"\nEducation Levels in this cluster:")
            level_counts = cluster_data['level'].value_counts()
            for level, count in level_counts.items():
                print(f"  - {level}: {count} programs")
        
        if 'field' in cluster_data.columns:
            print(f"\nFields of Study in this cluster:")
            field_counts = cluster_data['field'].value_counts()
            for field, count in field_counts.head(10).items():
                print(f"  - {field}: {count} programs")
        
        if 'geo' in cluster_data.columns:
            print(f"\nGeographies in this cluster:")
            geo_counts = cluster_data['geo'].value_counts()
            for geo, count in geo_counts.items():
                print(f"  - {geo}: {count} programs")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("CROSS-CLUSTER COMPARISON")
    print("=" * 80)
    
    if 'field' in analysis.columns:
        print("\nWhich fields appear in which clusters?")
        print("-" * 80)
        
        # Create a pivot showing fields by cluster
        field_cluster = pd.crosstab(analysis['field'], analysis['cluster'])
        print(field_cluster.head(20).to_string())
    
    if 'level' in analysis.columns:
        print("\n\nWhich education levels appear in which clusters?")
        print("-" * 80)
        
        level_cluster = pd.crosstab(analysis['level'], analysis['cluster'])
        print(level_cluster.to_string())
    
    # Save detailed breakdown
    output_dir = exemplars_path.parent
    output_file = output_dir / "cluster_analysis_detailed.csv"
    analysis.to_csv(output_file, index=False)
    print(f"\n\nDetailed analysis saved to: {output_file.resolve()}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
Based on the fields and programs in each cluster, you can now name them:

Examples:
- If a cluster has mostly Engineering, Computer Science → "High-earning STEM fields"
- If a cluster has mostly Arts, Humanities → "Liberal arts fields"
- If a cluster has mostly Health, Education → "Professional/service fields"

Look at the outcome profile plots (retraining_cluster_X_profile.png) alongside
these program lists to understand WHY each cluster has its particular outcomes.
    """)
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())