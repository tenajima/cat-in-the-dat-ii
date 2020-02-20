import gokart
import luigi
import pandas as pd


class GetDataSet(gokart.TaskOnKart):
    nrows = luigi.IntParameter()
    random_state = luigi.IntParameter()

    def run(self):
        train: pd.DataFrame = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
        test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

        if self.nrows:
            train = train.sample(n=self.nrows, random_state=self.random_state)

        dataset = pd.concat([train, test])
        self.dump(dataset)
