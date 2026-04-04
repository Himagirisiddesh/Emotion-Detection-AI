import argparse
import json

from emotion_utils import EmotionRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run face-aware emotion prediction on a single image.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full prediction payload as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    recognizer = EmotionRecognizer()
    analysis = recognizer.analyze_image(args.image)

    if args.json:
        serializable_analysis = {key: value for key, value in analysis.items() if key != "annotated_image"}
        print(json.dumps(serializable_analysis, indent=2))
        return

    if not analysis["success"]:
        print(analysis["message"])
        return

    print(analysis["message"])
    for index, detection in enumerate(analysis["detections"], start=1):
        print(f"Face {index}: {detection['result_text']}")


if __name__ == "__main__":
    main()
