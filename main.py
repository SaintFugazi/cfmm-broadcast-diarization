import argparse
from pipeline.orchestrator import DiarizationPipeline, STEPS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Diarization Pipeline")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Input CSV filename (without extension)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process (optional)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        choices=STEPS,
        metavar="STEP",
        help=f"Stop after this step. Choices: {', '.join(STEPS)}"
    )

    args = parser.parse_args()

    diarize = DiarizationPipeline(str(args.filename), limit=args.limit, stop_after=args.test)

    diarize.run()