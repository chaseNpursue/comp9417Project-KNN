import numpy as np
import csv

# k 
k = 5

# Attribute mappings for Automobile dataset
def getDict(list_attr):
    nums = range(len(list_attr))
    return dict(zip(list_attr, nums))

make = getDict(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 
        'mercedes-benz', 'mercury', 'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche',
        'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'])

fuel_type = getDict(['diesel', 'gas'])
aspiration = getDict(['std', 'turbo'])
num_doors = getDict(['four', 'two'])
num_doors['?'] = -1

body_style = getDict(['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible'])

drive_wheels = getDict(['4wd', 'fwd', 'rwd'])

engine_location = getDict(['front', 'rear'])
engine_type = getDict(['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor'])
num_cylinders = getDict(['eight', 'five', 'four', 'six', 'three', 'twelve', 'two'])
fuel_system = getDict(['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi'])

# Unpack data from ionosphere data by UCI https://archive.ics.uci.edu/ml/datasets/ionosphere
def getIonosphereData():
    data = []
    with open('ionosphere.data') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if row[34] == 'g':
                row[34] = 1
            elif row[34] == 'b':
                row[34] =  0
            else:
                print("Unknown label encountered while parsing dataset")
                exit(1)
            data.append(np.asarray(row, dtype=float))

    data = np.array(data)
    return data[:,:-1], data[:, -1]

# Unpack data from automobile data by UCI https://archive.ics.uci.edu/ml/datasets/automobile
def getAutomobileData(noncontinuous=False):
    data = []
    
    with open('imports-85.data') as file:
        csv_reader = csv.reader(file, delimiter=',')
        if noncontinuous == True:
            for row in csv_reader:
                add = True
                row[2] = make[row[2]] 
                row[3] = fuel_type[row[3]]
                row[4] = aspiration[row[4]]
                row[5] = num_doors[row[5]]
                row[6] = body_style[row[6]]
                row[7] = drive_wheels[row[7]]
                row[8] = engine_location[row[8]]
                row[14] = engine_type[row[14]]
                row[15] = num_cylinders[row[15]]
                row[17] = fuel_system[row[17]]

                for i in range(len(row)):
                    if row[i] == '?':
                        add = False

                if add == True:
                    data.append(np.asarray(row, dtype=float))
                    
        elif noncontinuous == False:
            for row in csv_reader:
                add = True
                row = [row[0], row[1], row[9], row[10], row[11], row[12], row[13], row[16], row[18],
                    row[19], row[20], row[21], row[22], row[23], row[24], row[25]]
                
                for i in range(len(row)):
                    if row[i] == '?':
                        add = False

                if add == True:
                    data.append(np.asarray(row, dtype=float))
        else:
            print("Unknown parameter")
            exit(1)
            
    data = np.array(data)
    return data[:,:-1], data[:, -1]

def getData(dataset):
    if dataset == 'Ionosphere': 
        return getIonosphereData()
    elif dataset == 'Automobile':
        return getAutomobileData()
    else:
        print("Unknown Dataset Name")
        exit(1)

# https://math.stackexchange.com/questions/139600/how-do-i-calculate-euclidean-and-manhattan-distance-by-hand

# This function returns the pairwise manhattan distance of 2 points in n dimensions(formula on the link above)
def manhattan_distance(x1, x2):
    return np.abs((x1 - x2)).sum()
    
# This function returns the pairwise euclidean distance of 2 points in n dimensions(formula on the link above)
def euclidean_distance(x1, x2):
    return np.sqrt(np.square((x1 - x2)).sum())

def calculate_weight(distance):
    return 1 / np.square(distance)

# Helper function to get first element of a list to use for sorting
def getFirstElement(val):
    return val[0]

# Pseudocode for algorithm in the following website 
# https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/
def classification(X_train, Y_train, X_test, k=5, distance='Euclidean', weighted=True):
    vals = []
    for i in range(len(X_train)):
        if distance == 'Euclidean':
            d = euclidean_distance(X_train[i], X_test)
        elif distance == 'Manhattan':
            d = manhattan_distance(X_train[i], X_test)
        else:
            print("Unknown distance")
            exit(1)
            
        if weighted==True:
            w = calculate_weight(d)
            vals.append([w, Y_train[i]])
        else: 
            vals.append([d, Y_train[i]])

        
    vals.sort(key=getFirstElement)
    preds = np.asarray(vals)[:k, -1] 
    unique_elem, freq = np.unique(preds, return_counts=True)

    return unique_elem[freq.argmax()]

def regression(x_train, y_train, test_data_final_x, k=5, distance='Euclidean', weighted='True'):
    numeric_predictions = []
    distances = []
    for i in range(len(x_train)):
        if distance == 'Euclidean':
            d = euclidean_distance(x_train[i],test_data_final_x)
            distances.append(d)
        else:
            d = manhattan_distance(x_train[i],test_data_final_x)
            distances.append(d)

    sorted_index_distances = np.argsort(distances)
    #print(sorted_index_distances)
    final_weight = 0
    total_y = 0
    for i in sorted_index_distances[:k]:
        #only calculate weight if needed
        if weighted:
            if distances[i] == 0:
                weight = 1
            else:
                weight = calculate_weight(distances[i])
            #print('distance:',distances[i])
            #print('weight', weight)
            #print(distances[i] == 0)
            total_y += y_train[i] * weight
            final_weight += weight
            #print(distances.index(distances[i] == 0))]

        else:
            total_y += y_train[i]
        #print(final_weight)
    numeric_predictions.append(total_y/final_weight if weighted else total_y/k)
    return numeric_predictions[0]

def knn(X_train, Y_train, X_test, k=5, _type='classification', distance='Euclidean', weighted=True):
    if _type == 'classification':
        return classification(X_train, Y_train, X_test, k, distance, weighted)
    elif _type == 'regression':
        return regression(X_train, Y_train, X_test, k, distance, weighted)
    else: 
        print("Invalid operation type")
        exit(1)
        
def loocv(dataset='Ionosphere', _type='classification', k=5, distance='Euclidean', weighted=True):
    data_X, data_Y = getData(dataset)

    err = 0
    for i in range(len(data_X)):
        x_in = np.concatenate((data_X[:i], data_X[i + 1:]))
        y_in = np.concatenate((data_Y[:i], data_Y[i + 1:]))
        x_out = data_X[i]
        y_out = knn(x_in, y_in, x_out, k=5,_type='classification', distance='Euclidean', weighted=True)
        err += np.square(data_Y[i] - y_out)
        
    err = err/len(data_X)    

    print("Error by Leave One Out Cross Validation is " + str(err))


if __name__ == "__main__":
    loocv('Automobile', 'regression', 5, 'Euclidean', True)
