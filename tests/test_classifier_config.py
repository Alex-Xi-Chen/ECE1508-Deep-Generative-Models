from musemotion.training.classifier import apply_split_sample_limits


class FakeSplit:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def select(self, indices):
        return FakeSplit([self.values[index] for index in indices])


def test_apply_split_sample_limits_uses_split_specific_counts():
    dataset = {
        "train": FakeSplit(range(5)),
        "validation": FakeSplit(range(4)),
        "test": FakeSplit(range(3)),
    }

    limited = apply_split_sample_limits(
        dataset,
        {
            "max_train_samples": 2,
            "max_validation_samples": 1,
            "max_test_samples": 2,
        },
    )

    assert len(limited["train"]) == 2
    assert len(limited["validation"]) == 1
    assert len(limited["test"]) == 2
