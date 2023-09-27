import datasets as ds
import pandas as pd

_CITATION = """\
@article{saharia2022photorealistic,
  title={Photorealistic text-to-image diffusion models with deep language understanding},
  author={Saharia, Chitwan and Chan, William and Saxena, Saurabh and Li, Lala and Whang, Jay and Denton, Emily L and Ghasemipour, Kamyar and Gontijo Lopes, Raphael and Karagol Ayan, Burcu and Salimans, Tim and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={36479--36494},
  year={2022}
}
"""

_SHORT_DESCRIPTION = "DrawBench is a new comprehensive and challenging evaluation benchmark for the text-to-image task."

_DESCRIPTION = """\
DrawBench is a comprehensive and challenging set of prompts that support the evaluation and comparison of text-to-image models. This benchmark contains 11 categories of prompts, testing different capabilities of models such as the ability to faithfully render different colors, numbers of objects, spatial relations, text in the scene, and unusual interactions between objects.\
"""

_HOMEPAGE = "https://imagen.research.google/"

_URL = "https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/gviz/tq?tqx=out:csv"
_CATEGORIES = [
    "Colors",
    "Conflicting",
    "Counting",
    "DALL-E",
    "Descriptions",
    "Gary Marcus et al.",
    "Misspellings",
    "Positional",
    "Rare Words",
    "Reddit",
    "Text",
]


class DrawBench(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            version=VERSION,
            description=_SHORT_DESCRIPTION,
        ),
    ]

    def _info(self):
        features = ds.Features(
            {
                "prompts": ds.Value("string"),
                "category": ds.ClassLabel(num_classes=11, names=_CATEGORIES),
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        file_path = dl_manager.download(_URL)
        return [
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={"file_path": file_path},
            )
        ]

    def _generate_examples(self, file_path: str):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()

        for i, example in enumerate(df.to_dict(orient="records")):
            yield i, example
