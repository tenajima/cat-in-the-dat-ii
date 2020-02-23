import luigi
from dotenv import load_dotenv

from scripts.train.null_importance import NullImportance

if __name__ == "__main__":
    load_dotenv()
    # luigi.build([NullImportance()], local_scheduler=True)
    luigi.run(["NullImportance"])
