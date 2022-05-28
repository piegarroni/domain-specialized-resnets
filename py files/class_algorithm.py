
from attr import define
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class algorithm():
        def __init__(self, n_clusters, data):
            self.n_clusters=n_clusters
            self.data=data
            self.data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

        def domain_selection(self):
            """
            Method that trains model to understand to which cluster image belong to without knowing the category
            """
            # Define model
            model = Sequential()
            model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
            model.add(Dense(self.n_clusters, activation='softmax', name='predictions'))
            model.layers[0].trainable = False

            model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

            image_size = 224

            train_generator = self.data_generator.flow_from_directory(
                    r'/mnt/c/Users/pietr/Desktop/ImageNet/domain/train',
                    target_size=(image_size, image_size),
                    batch_size=24,
                    class_mode='categorical')

            validation_generator = self.data_generator.flow_from_directory(
                    r'/mnt/c/Users/pietr/Desktop/ImageNet/domain/test_val',
                    target_size=(image_size, image_size),
                    class_mode='categorical')

            model.fit(train_generator, validation_generator, epochs=25)

            model.predict(self.data)

            self.divide_cluster()
            self.domain_model=model

            # generate csv file with class belonging

        def predict_cluster(self):
            self.domain_model.predict(train_generator)
            self.domain_model.predict(val_generator)
            self.domain_model.predict(test_generator)

            for i in prediction:
                create new repository (named after the cluster)
                define
                compile
                fit




            # get an array like with predictions

'''
        def divide_cluster():
            """
            Method that puts each category folder into cluster folder 
            """
            TRAIN_DATA_FOLDER = "/mnt/c/Users/pietr/Desktop/ImageNet/Image_class/train/"
            valPaths=[]
            Clusters=[]
            which_cluster=pd.read_csv(r'/mnt/c/Users/pietr/Desktop/ImageNet/domain_belonging.csv')    #### get csv with PredictionCode and Cluster

            for i, clus in enumerate(which_cluster['Cluster']):
                valPaths.append(TRAIN_DATA_FOLDER+"/"+clus)
                Clusters.append(which_cluster['Cluster'][i])

           # print(valPaths[:10])
           # print(Clusters[:10])

            for valPath, cluster in zip(valPaths, Clusters):
                if not os.path.isdir(TRAIN_DATA_FOLDER + str(cluster)):
                    os.mkdir(TRAIN_DATA_FOLDER + str(cluster))
                try:
                    os.rename(valPath, TRAIN_DATA_FOLDER + str(cluster) +'/' + valPath.split("/")[-1])
                except FileNotFoundError:
                    pass
'''

        def classification(self):
            """
            Method to define and train a classification model for each cluster
            """

            # Define model
 

            image_size = 224
            data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
            scores=[]
            for cluster_i in 'list of unique clusters':
                model = Sequential()
                model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
                model.add(Dense(len(cluster_i), activation='softmax', name='predictions'))
                model.layers[0].trainable = False

                model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])
            
                # Define data generator
                train_generator = data_generator.flow_from_directory(
                        r'/mnt/c/Users/pietr/Desktop/ImageNet/train/'+cluster_i,
                        target_size=(image_size, image_size),
                        batch_size=24,
                        class_mode='categorical')

                validation_generator = data_generator.flow_from_directory(
                        r'/mnt/c/Users/pietr/Desktop/ImageNet/val/'cluster_i,
                        target_size=(image_size, image_size),
                        class_mode='categorical')

                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

                # Fit model
                model.fit(
                    train_generator,
                    steps_per_epoch = train_generator.samples // 48,
                    validation_data = validation_generator, 
                    validation_steps = validation_generator.samples // 48,
                    callbacks=[callback],
                    epochs = 15)
                
                accuracy, robustness =self.evaluate(model)
                scores.append(accuracy, robustness)

            return scores

        def evaluate(self, model):
            """
            Method that evaluates the accuracy and robustness of each model
            """

            image_size = 224

            test_generator_accuracy = self.data_generator.flow_from_directory(
                    r'/mnt/c/Users/pietr/Desktop/ImageNet/test/test',
                    target_size=(image_size, image_size),
                    class_mode='categorical')

            test_generator_robustness = self.data_generator.flow_from_directory(
                    r'/mnt/c/Users/pietr/Desktop/ImageNet/other_data',
                    target_size=(image_size, image_size),
                    class_mode='categorical')        

            accur=model.evaluate_generator(generator=test_generator_accuracy)
            print("Accuracy ResNet TL= ", accur[1])
          

            robus=model.evaluate_generator(generator=test_generator_robustness)
            print("Robustness ResNet TL= ", robus[1])
            return(accur, robus)


##################################################################################


# try to fit models to cluster without domain selection first

# second step would be to create other model first (reccomended)