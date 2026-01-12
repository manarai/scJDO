
import argparse, json, os
from scqdiff.viz.plot_archetypes import plot_archetype_heatmaps, plot_temporal_activations, plot_singular_values
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prefix', required=True); ap.add_argument('--max-archetypes', type=int, default=3); ap.add_argument('--save', action='store_true'); args=ap.parse_args()
    patt=f"{args.prefix}.patterns.npy"; Ut=f"{args.prefix}.U_t.npy"; Sv=f"{args.prefix}.S.npy"; meta=f"{args.prefix}.meta.json"
    edges=None
    if os.path.exists(meta): 
        with open(meta,'r',encoding='utf-8') as f: m=json.load(f); edges=m.get('edges', None)
    save_prefix=args.prefix if args.save else None
    plot_archetype_heatmaps(patt, max_archetypes=args.max_archetypes, save_prefix=save_prefix)
    plot_temporal_activations(Ut, edges_meta=edges, save_path=(f"{args.prefix}.U_t.png" if args.save else None))
    plot_singular_values(Sv, save_path=(f"{args.prefix}.S.png" if args.save else None))
if __name__=='__main__': main()
