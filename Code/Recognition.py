import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

class Recognition:
    def __init__(self, original, path2, path3, path4, path5, path6):
        self.original = cv2.imread(original, 0)
        self.hair = cv2.imread(path2, 0)
        self.glasses = cv2.imread(path3, 0)
        self.chubby = cv2.imread(path4, 0)
        self.beard = cv2.imread(path5, 0)
        self.other = cv2.imread(path6, 0)

        self.width, self.height = self.original.shape

    def visualize_faces(self, all_faces, figure_title):
        plt.figure(figure_title, figsize=(12, 5))
        titles = ['Original', 'Hair', 'Glasses', 'Chubby', 'Beard', 'Other']

        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(all_faces[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

    def eigenfaces(self, all_faces, name):
        # create another matrix with each row as the flattened image
        data = np.zeros((len(all_faces), self.width*self.height), dtype=np.float32)
        index = 0
        for face in all_faces:
            face = face.flatten()
            data[index, :] = face
            index += 1

        # calculate the mean and eigenvectors
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data, mean=None, maxComponents=10)
        eigenfaces = []

        for vec in eigenvectors:
            eigenfaces.append(vec.reshape(self.width, self.height))

        plt.figure('Eigenfaces for ' + name, figsize=(12, 5))
        titles = ['Original', 'Hair', 'Glasses', 'Chubby', 'Beard', 'Other']

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(eigenfaces[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        return eigenfaces, mean, eigenvalues

    def reconstruct(self, eigenfaces, mean, name, number_of_eigenfaces, alpha=0.03):
        # to reconstruct the face, we add an eigenface to the mean
        new_face = np.zeros((self.width, self.height), dtype=np.float32)
        # get the sum of the scalar multiplier and the eigenface/s
        for i in range(len(eigenfaces)):
            new_face += alpha * eigenfaces[i]

        # add the above sum to the mean to get the reconstructed face
        mean = mean.reshape(self.width, self.height)
        new_face += mean

        title = 'Reconstructed Face for ' + name + ' with ' + str(number_of_eigenfaces) + ' eigenface/s'
        plt.figure(title, figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.original, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(new_face.astype(np.uint8), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')

    def reconstruct_with_other_eigenface(self, eigenfaces, eigenvalue, mean, name, other, alpha=0.07):
        # to reconstruct the face, we add an eigenface to the mean
        new_face = np.zeros((self.width, self.height), dtype=np.float32)

        # get the highest eigenvalue
        max_eigenvalue = max(eigenvalue)

        # get the index of the highest eigenvalue
        max_eigenvalue_index = (np.where(eigenvalue == max_eigenvalue))[0][0]
        # get the corresponding eigenface with that eigenvalues index
        max_eigenface = eigenfaces[max_eigenvalue_index]

        # get the sum of the scalar multiplier and the eigenface
        new_face += alpha * max_eigenface

        # add the above sum to the mean to get the reconstructed face
        mean[max_eigenvalue_index] = mean[max_eigenvalue_index].reshape(self.width, self.height)
        new_face += mean[max_eigenvalue_index]

        title = 'Reconstructed Face for ' + name + ' with ' + other + '\'s eigenface'
        plt.figure(title, figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.original, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(new_face, cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')


# setting the paths
kid_path = 'images/Faces/kid.jpeg'
kid_hair_path = 'images/Faces/kid_with_hair.jpeg'
kid_glasses_path = 'images/Faces/kid_with_glasses.jpeg'
kid_chubby_path = 'images/Faces/kid_chubby.jpeg'
kid_beard_path = 'images/Faces/kid_beard.jpeg'
kid_other_path = 'images/Faces/kid_nose.jpeg'

old_path = 'images/Faces/old.jpeg'
old_hair_path = 'images/Faces/hair.jpeg'
old_glasses_path = 'images/Faces/glasses.jpeg'
old_chubby_path = 'images/Faces/chubby.jpeg'
old_beard_path = 'images/Faces/beard.jpeg'
old_other_path = 'images/Faces/nose.jpeg'

woman_path = 'images/Faces/woman.jpeg'
woman_with_hair = 'images/Faces/woman_with_hair.jpeg'
woman_with_glasses = 'images/Faces/woman_chubby.jpeg'
woman_chubby = 'images/Faces/woman_beard.jpeg'
woman_with_beard = 'images/Faces/woman_beard.jpeg'
woman_other = 'images/Faces/woman_freckles.jpeg'

# create a kid face object
kid = Recognition(kid_path, kid_hair_path, kid_glasses_path, kid_chubby_path, kid_beard_path, kid_other_path)
# kid.visualize_faces([kid.original, kid.hair, kid.glasses, kid.chubby, kid.beard, kid.other], 'All Original Faces')
kid_faces = [kid.original, kid.hair, kid.glasses, kid.chubby, kid.beard, kid.other]
eigenfaces_kid, mean_kid, eigenvalues_kid = kid.eigenfaces(kid_faces, 'Kid')
kid.visualize_faces(eigenfaces_kid, 'Eigenfaces for Kid')
# reconstruct with 1 random eigenface from the set of eigenfaces
x = random.randint(0, 5)
print(x)
kid.reconstruct(eigenfaces_kid[x], mean_kid, 'Kid', 1)
# reconstruct with 6 eigenfaces
kid.reconstruct(eigenfaces_kid, mean_kid, 'Kid', len(eigenfaces_kid))

# create an old face object
old = Recognition(old_path, old_hair_path, old_glasses_path, old_chubby_path, old_beard_path, old_other_path)
old_faces = [old.original, old.hair, old.glasses, old.chubby, old.beard, old.other]
eigenfaces_old, mean_old, eigenvalues_old = old.eigenfaces(old_faces, 'Old Man')
# reconstruct with 1 random eigenface
old.reconstruct(eigenfaces_old[x], mean_old, 'Old', 1)
# reconstruct with 6 eigenfaces
old.reconstruct(eigenvalues_old, mean_old, 'Old', 6)

# create a woman face object
woman = Recognition(woman_path, woman_with_hair, woman_with_glasses, woman_chubby, woman_with_beard, woman_other)
woman_faces = [woman.original, woman.hair, woman.glasses, woman.chubby, woman.beard, woman.other]
eigenfaces_woman, mean_woman, eigenvalues_woman = woman.eigenfaces(woman_faces, 'Woman')

# reconstruct with 1 random eigenface
woman.reconstruct(eigenfaces_woman[x], mean_woman, 'Woman', 1)

# reconstruct with 6 eigenfaces
woman.reconstruct(eigenfaces_woman, mean_woman, 'Woman', 6)

# reconstruct with a different eigenface (kid in this case)
#kid.reconstruct_with_other_eigenface(eigenfaces_old, eigenvalues_kid, eigenfaces_kid, 'Kid', 'Old Man')
#woman.reconstruct_with_other_eigenface(eigenfaces_old, eigenvalues_woman, eigenfaces_woman, 'Woman', 'Old Man')
plt.show()
