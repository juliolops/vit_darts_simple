# moq-nas/scripts/fairness_baseline/prepare_data.py

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from core.fairness.build import (
    build_facet_csv,
    build_face_binary_dataset,
    build_person_binary_dataset,
    build_square_resized_version,
)

def main():
    """
    Main script to orchestrate the construction of all datasets
    required for the fairness analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Prepares all datasets for fairness training and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Arguments for the PERSON dataset (from COCO) ---
    parser.add_argument(
        '--build_person', action='store_true',
        help="Enables the build process for the PERSON/NON_PERSON dataset."
    )
    parser.add_argument(
        '--coco_root', type=str, default='datasets/COCO_sub',
        help="Path to the COCO root directory (containing train2017, val2017, annotations)."
    )
    parser.add_argument(
        '--person_out_dir', type=str, default='datasets/personbin_data',
        help="Output directory for the PERSON/NON_PERSON dataset."
    )

    # --- Arguments for the FACE dataset (from WIDER) ---
    parser.add_argument(
        '--build_face', action='store_true',
        help="Enables the build process for the FACE/NON_FACE dataset."
    )
    parser.add_argument(
        '--wider_root', type=str, default='datasets/WIDER',
        help="Path to the WIDER Face root directory."
    )
    parser.add_argument(
        '--places_root', type=str, default='datasets/PLACES365/val',
        help="Path to a directory of negative images (e.g., Places365) for the face dataset."
    )
    parser.add_argument(
        '--face_out_dir', type=str, default='datasets/facebin_data',
        help="Output directory for the FACE/NON_FACE dataset."
    )

    # --- Arguments for the FACET CSV ---
    parser.add_argument(
        '--build_facet', action='store_true',
        help="Enables the build process for the FACET evaluation CSV."
    )
    parser.add_argument(
        '--facet_ann', type=str, default='datasets/facet_data/annotations/annotations.csv',
        help="Path to the raw FACET annotations CSV file."
    )
    parser.add_argument(
        '--facet_img_dirs', nargs='+', default=['datasets/facet_data/imgs_1', 'datasets/facet_data/imgs_2'],
        help="A list of directories containing the FACET images."
    )
    parser.add_argument(
        '--facet_out_csv', type=str, default='datasets/facet_data/facet_eval.csv',
        help="Output path for the processed FACET CSV."
    )

    # --- Arguments for square-resized mirrors ---
    parser.add_argument(
        '--make_person_resized', action='store_true',
        help="Generate a square resized mirror of the PERSON dataset."
    )
    parser.add_argument(
        '--make_face_resized', action='store_true',
        help="Generate a square resized mirror of the FACE dataset."
    )
    parser.add_argument(
        '--resize_target', type=int, default=96,
        help="Target side length for the resized mirror (e.g., 48, 64, 96, 128)."
    )
    parser.add_argument(
        '--resize_mode', type=str, default='letterbox', choices=['center_crop', 'letterbox'],
        help="Square strategy: crop center or pad (letterbox)."
    )
    parser.add_argument(
        '--jpg_quality', type=int, default=90,
        help="JPEG quality for resized images (1-100)."
    )
    parser.add_argument(
        '--person_resized_out_dir', type=str, default=None,
        help="Output dir for the resized PERSON mirror. "
             "Defaults to <person_out_dir>_<resize_target>."
    )
    parser.add_argument(
        '--face_resized_out_dir', type=str, default=None,
        help="Output dir for the resized FACE mirror. "
             "Defaults to <face_out_dir>_<resize_target>."
    )
    parser.add_argument(
        '--person_src_for_resize', type=str, default=None,
        help="Custom source root for PERSON resizing (defaults to --person_out_dir)."
    )
    parser.add_argument(
        '--face_src_for_resize', type=str, default=None,
        help="Custom source root for FACE resizing (defaults to --face_out_dir)."
    )

    args = parser.parse_args()

    any_action = any([
        args.build_person, args.build_face, args.build_facet,
        args.make_person_resized, args.make_face_resized,
    ])
    if not any_action:
        print("\nNo build action was specified. Use --build_person, --build_face, "
              "--build_facet, --make_person_resized, or --make_face_resized.")
        print("Example: python scripts/fairness_baseline/prepare_data.py "
              "--make_person_resized --resize_target 48")
        return

    if args.build_person:
        print("\n--- Starting PERSON/NON_PERSON dataset build ---")
        build_person_binary_dataset(
            coco_root=args.coco_root,
            out_dir=args.person_out_dir,
        )
        print("--- Finished: PERSON/NON_PERSON ---")

    if args.build_face:
        print("\n--- Starting FACE/NON_FACE dataset build ---")
        build_face_binary_dataset(
            wider_root=args.wider_root,
            neg_root=args.places_root,
            out_dir=args.face_out_dir,
        )
        print("--- Finished: FACE/NON_FACE ---")

    if args.build_facet:
        print("\n--- Starting FACET CSV build ---")
        build_facet_csv(
            ann_csv=args.facet_ann,
            img_dirs=args.facet_img_dirs,
            out_csv=args.facet_out_csv,
        )
        print("--- Finished: FACET CSV ---")

    if args.make_person_resized:
        src = args.person_src_for_resize or args.person_out_dir
        dst = args.person_resized_out_dir or f"{args.person_out_dir}_{args.resize_target}"
        print(f"\n--- Making PERSON {args.resize_target}x{args.resize_target} mirror ---")
        build_square_resized_version(
            src_root=src,
            dst_root=dst,
            target=args.resize_target,
            mode=args.resize_mode,
            quality=args.jpg_quality,
        )
        print(f"--- Finished: PERSON {args.resize_target}x{args.resize_target} mirror → {dst} ---")

    if args.make_face_resized:
        src = args.face_src_for_resize or args.face_out_dir
        dst = args.face_resized_out_dir or f"{args.face_out_dir}_{args.resize_target}"
        print(f"\n--- Making FACE {args.resize_target}x{args.resize_target} mirror ---")
        build_square_resized_version(
            src_root=src,
            dst_root=dst,
            target=args.resize_target,
            mode=args.resize_mode,
            quality=args.jpg_quality,
        )
        print(f"--- Finished: FACE {args.resize_target}x{args.resize_target} mirror → {dst} ---")


if __name__ == "__main__":
    main()
