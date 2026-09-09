"""Print a concise parameter report for the specified SeaGlass model."""

from .transformer import SeaGlassTransformer


def main() -> None:
    model = SeaGlassTransformer()
    target = 41_600_000
    count = model.parameter_count
    print("SeaGlass Transformer parameter report")
    print(f"Parameters: {count:,} ({count / 1_000_000:.2f}M)")
    print(f"Target:     {target:,} ({target / 1_000_000:.1f}M)")
    print(f"Difference: {count - target:+,}")
    print(f"Tied embeddings: {model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()}")


if __name__ == "__main__":
    main()
