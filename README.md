# Projects - Couple of interesting notebooks

Project 1 - Face recognition
According to Yan, Kriegman, and Ahuja, face detection can be categorized into feature-based, appearance-based, knowledge-based and template matching. As the name suggests, feature-based relies on structural features of the face, whereas knowledge-based relies on the pre-existing knowledge and rules pertaining to the face constitution. Template matching correlates the input images with pre-defined, parametrized templates whereas appearance-based banks on a training set to generate face models.

The last method (appearance-based) has superior performance as it involves statistical analysis and machine learning. Another application of this method would be face feature extraction.

Sub-methods of Appearance-based methodology
Following are the sub-methods and we are planning to employ Eigen-based method and SVM, among others, for our analysis.

Eigenface-based method:- has been around since 1991 and uses Principal Component Analysis (PCA) to efficiently represent faces. PCA happens to be the linear dimensionality reduction using approximated Singular Value Decomposition of the data and keeping only the most significant singular vectors to project the data to a lower dimensional space.

Distribution-based method:- The algorithms like PCA and Fisher’s Discriminant can be used to define the subspace representing facial patterns. There is a trained classifier, which correctly identifies instances of the target pattern class from the background image patterns.

Neural-Networks:- Many detection problems like object detection, face detection, emotion detection, and face recognition, etc. have been faced successfully by Neural Networks.

Support Vector Machine (SVM):- Support Vector Machines are linear classifiers that maximise the margin between the decision hyperplane and the examples in the training set. Osuna et al. first applied this classifier to face detection.

Sparse Network of Winnows:- They defined a sparse network of two linear units or target nodes; one represents face patterns and other for the non-face patterns. It is less time consuming and efficient.

Naive Bayes Classifiers:- They computed the probability of a face to be present in the picture by counting the frequency of occurrence of a series of the pattern over the training images. The classifier captured the joint statistics of local appearance and position of the faces.

Hidden Markov Model:- The states of the model would be the facial features, which usually described as strips of pixels. HMM’s commonly used along with other methods to build detection algorithms.
