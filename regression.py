import numpy as np


class KNN:

    def __init__(self, k, weighted=False, distance='Euclidean'):
        self.k = k
        self.weighted = weighted
        self.distance = distance

    @staticmethod
    def automobile_datapreprocessing(self):

        file = open("imports-85.data")
        lines = file.readlines()
        useful_data = []
        useful_data_final_x = []
        useful_data_final_y = []
        for line in lines:
            if '?' not in line:
                split_data = line.split(",")

                useful_data.append(split_data[1])
                useful_data.append(split_data[9])
                useful_data.append(split_data[10])
                useful_data.append(split_data[11])
                useful_data.append(split_data[12])
                useful_data.append(split_data[13])
                useful_data.append(split_data[16])
                useful_data.append(split_data[18])
                useful_data.append(split_data[19])
                useful_data.append(split_data[20])
                useful_data.append(split_data[21])
                useful_data.append(split_data[22])
                useful_data.append(split_data[23])
                useful_data.append(split_data[24])
                useful_data.append(split_data[0])
                useful_data.append(split_data[25])
                useful_data_new = [float(x) for x in useful_data]
                useful_data_final_x.append(useful_data_new[:-1])
                useful_data_final_y.append(useful_data_new[-1])
                useful_data.clear()
        return useful_data_final_x, useful_data_final_y

    @staticmethod
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.square((x1 - x2)).sum())

    @staticmethod
    def manhattan_distance(self, x1, x2):
        return np.abs((x1 - x2)).sum()

    @staticmethod
    def caculate_weight(self, distance):
        return 1 / np.square(distance)

    @staticmethod
    def numeric_predict(self, x_train, y_train, test_data_final_x):
        numeric_predictions = []
        distances = []
        for i in range(len(x_train)):
            if self.distance:
                d = self.euclidean_distance(self,x_train[i],test_data_final_x)
                distances.append(d)
            else:
                d = self.manhattan_distance(self,x_train[i],test_data_final_x)
                distances.append(d)

        sorted_index_distances = np.argsort(distances)
        #print(sorted_index_distances)
        final_weight = 0
        total_y = 0
        for i in sorted_index_distances[:self.k]:
            #only calculate weight if needed
            if self.weighted:
                if distances[i] == 0:
                    weight = 1
                else:
                    weight = self.caculate_weight(self, distances[i])
                #print('distance:',distances[i])
                #print('weight', weight)
                #print(distances[i] == 0)
                total_y += y_train[i] * weight
                final_weight += weight
                #print(distances.index(distances[i] == 0))]

            else:
                total_y += y_train[i]
            #print(final_weight)
        numeric_predictions.append(total_y/final_weight if self.weighted else total_y/self.k)
        return numeric_predictions[0]

    @staticmethod
    def loocv(self):
        train_x, train_y = self.automobile_datapreprocessing(self)
        train_x_final = np.array(train_x)
        train_y_final = np.array(train_y)
        error = 0
        for i in range(len(train_x_final)):
            x_in = np.concatenate((train_x_final[:i], train_x_final[i+1:]))
            y_in = np.concatenate((train_y_final[:i], train_y_final[i+1:]))
            x_out = train_x_final[i]
            y_out = self.numeric_predict(self, x_in, y_in, x_out)
            #print(train_y_final[i] - y_out)
            error += np.square(train_y_final[i] - y_out)
        error = error / len(train_x_final)
        print(error)


if __name__ == "__main__":
    classifier = KNN(k=7, weighted=True)
    classifier.loocv(classifier)

    #test_error = mean_squared_error(y_test, pred_test)
