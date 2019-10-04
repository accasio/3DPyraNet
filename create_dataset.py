import glob
import numpy as np


def main():
    x = []
    y = []

    for split in ["test", "train"]:
        for file in sorted(glob.glob("dataset/numpys/%s/*" % split)):
            if "Beach" in file:
                y.append(0)
            elif "BuildingCollapse" in file:
                y.append(1)
            elif "Elevator" in file:
                y.append(2)
            elif "Escalator" in file:
                y.append(3)
            elif "FallingTree" in file:
                y.append(4)
            elif "Fireworks" in file:
                y.append(5)
            elif "ForestFire" in file:
                y.append(6)
            elif "Fountain" in file:
                y.append(7)
            elif "Highway" in file:
                y.append(8)
            elif "LightningStorm" in file:
                y.append(9)
            elif "Marathon" in file:
                y.append(10)
            elif "Ocean" in file:
                y.append(11)
            elif "Railway" in file:
                y.append(12)
            elif "RushingRiver" in file:
                y.append(13)
            elif "SkyClouds" in file:
                y.append(14)
            elif "Snowing" in file:
                y.append(15)
            elif "Street" in file:
                y.append(16)
            elif "Waterfall" in file:
                y.append(17)
            elif "WavingFlags" in file:
                y.append(18)
            elif "WindmillFarm" in file:
                y.append(19)

            train_file = np.load(file)

            x.append(train_file)

        if split == "test":
            np.save("dataset/yupp/TestVal.npy", np.asarray(x))
            np.save("dataset/yupp/TestVal_label.npy", np.asarray(y))

        else:
            np.save("dataset/yupp/Training.npy", np.asarray(x))
            np.save("dataset/yupp/Training_label.npy", np.asarray(y))


if __name__ == '__main__':
    main()
