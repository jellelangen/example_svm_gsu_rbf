import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tensorflow as tf
import random
import os
np.random.seed(4)
tf.random.set_seed(4)
random.seed(4)

def load_fashion_Mnist(num_samples=60000):
    
    train = np.loadtxt(r'C:\Users\jelle\Desktop\bachelor project\gitrepo\SVM-GSU\svm_gsu\data\train\fashion-mnist_train.csv', delimiter=',', skiprows=1, max_rows=num_samples)
    test = np.loadtxt(r'C:\Users\jelle\Desktop\bachelor project\gitrepo\SVM-GSU\svm_gsu\data\test\fashion-mnist_test.csv', delimiter=',', skiprows=1, max_rows=num_samples)
    samples_train = train[:, 1:]
    labels_train = train[:, 0]
    samples_test = test[:, 1:]
    labels_test = test[:, 0]

    return samples_train, labels_train, samples_test, labels_test

def select_samples_by_class(samples, labels, class1, class2, num_samples):
    all_samples = [
                    np.where(labels == 0)[0],
                    np.where(labels == 1)[0],
                    np.where(labels == 2)[0],
                    np.where(labels == 3)[0],
                    np.where(labels == 4)[0],
                    np.where(labels == 5)[0],
                    np.where(labels == 6)[0],
                    np.where(labels == 7)[0],
                    np.where(labels == 8)[0],
                    np.where(labels == 9)[0]
                        ]
    selected_class1_indices = np.random.choice(all_samples[class1], num_samples, replace=False)
    if class2 != 10:
        selected_class2_indices = np.random.choice(all_samples[class2], num_samples, replace=False)
    else:
        selected_class2_indices = np.concatenate(
            [all_samples[i][:num_samples//9] for i in range(10) if i != class1]
        )
    selected_samples = np.concatenate((samples[selected_class1_indices], samples[selected_class2_indices]), axis=0)
    selected_labels = np.concatenate((labels[selected_class1_indices], labels[selected_class2_indices]), axis=0)
    selected_labels[selected_labels == class1] = -1
    selected_labels[selected_labels != -1] = 1
    print(selected_labels)
    return selected_samples, selected_labels



def generate_circular_dataset(center_x=1, center_y=1, size=4, inner_diameter=0.5, outer_diamater=1):
    """
        This function creates a synthetic dataset with two classes. The first class is a circle with a radius of inner_diameter, and the second class is a circle with a radius of outer_diameter. The two circles are concentric.
    """	
    half_size = size // 2
    
    # sample angles from 0 to 2pi for both classes and diameters from 0 to inner_diameter for class_1 and from inner_diameter to outer_diameter for class_2
    
    inner_radii = np.full(half_size, inner_diameter)
    angles1 = np.linspace(0, 2*np.pi, half_size, endpoint=True)
    x_class_1 = inner_radii * np.cos(angles1) + np.full(len(angles1),center_x)
    y_class_1 = inner_radii * np.sin(angles1) + np.full(len(angles1),center_y) 

    outer_radii = np.full(half_size, outer_diamater)
    angles2 = np.linspace(0, 2*np.pi, half_size, endpoint=True)
    x_class_2 = outer_radii * np.cos(angles2) + np.full(len(angles2),center_x)
    y_class_2 = outer_radii * np.sin(angles2) + np.full(len(angles2),center_y)

    
    class_1 = np.column_stack((x_class_1, y_class_1))
    class_2 = np.column_stack((x_class_2, y_class_2))

    samples = np.vstack((class_1, class_2))
    labels = np.array([1]*half_size + [-1]*half_size)
    return samples, labels


def generate_non_linear_svm_toy_dataset(size=4, x_range=100, y_range=100):
    """
        This function creates 2d synthetic datasets

        Args:
            size: Total size of the dataset. There will be two classes each having size/2 number of samples
            x_range: The range in which the samples will be generated is (0, x_range), on the x-axis
            y_range: The range in which the samples will be generated is (0, y_range), on the y-axis
        Returns:
            A dataset of two classes with samples and their corresponding labels. 
    """
    half_size = size // 2
    x_class_1 = x_range * 0.25 #divide x_range in quarters: x coordinate class 1 is 25% of full x_range
    x_class_2 = x_range * 0.75 #x_coordinate of class 2 is 75% of

    #class 1
    y_values = np.linspace(0-y_range, y_range, size//2) #equally spaced samples between -y_range and y_range.
    x_values = np.linspace(-x_class_1, -x_class_1, size//2)
    class_1 = np.column_stack((x_values, y_values))

    #class 2
    y_values = np.linspace(0-y_range, y_range, size//2) #equally spaced samples between -y_range and y_range.
    x_values = np.linspace(x_class_2, x_class_2, size//2)
    class_2 = np.column_stack((x_values, y_values))

    samples = np.vstack((class_1, class_2)) #combine the samples into one array
    labels = np.array([-1,1,1,-1])
    return samples, labels

def generate_linear_svm_toy_dataset(size=6, x_range=100, y_range=100, x_range_variability=0.1):
    """
        This function creates 2d synthetic datasets

        Args:
            size: Total size of the dataset. There will be two classes each having size/2 number of samples
            x_range: The range in which the samples will be generated is (0, x_range), on the x-axis
            y_range: The range in which the samples will be generated is (0, y_range), on the y-axis
        Returns:
            A dataset of two classes with samples and their corresponding labels. 
    """
    half_size = size // 2
    x_class_1 = x_range * 0.25 #divide x_range in quarters: x coordinate class 1 is 25% of full x_range
    x_class_2 = x_range * 0.75 #x_coordinate of class 2 is 75% of full x_range
    #class 1
    y_values = np.linspace(0-y_range, y_range, size//2) #equally spaced samples between -y_range and y_range.
    x_values = np.linspace(-x_class_1+(x_range_variability*x_range), -x_class_1-(x_range_variability*x_range), size//2) 
    class_1 = np.column_stack((x_values, y_values))
    
    #class 2
    y_values = np.linspace(0-y_range, y_range, size//2) #equally spaced samples between -y_range and y_range.s
    x_values = np.linspace(x_class_2+(x_range_variability*x_range), x_class_2-(x_range_variability*x_range), size//2) 
    class_2 = np.column_stack((x_values, y_values))


    samples = np.vstack((class_1, class_2)) #combine the samples into one array
    labels = np.array([-1]*half_size + [1]*half_size)

    return samples, labels


def generate_covariance_matrices(samples,  covariance=[[1, 0], [0, 1]], increasing=False):
    """
        This function len(samples) 2x2 covariance matrices based on a provided covariance matrix. If increasing is set to true, the first half of the matrices will have increasing uncertainty,
        whereas the second half will have decreasing uncertainty.
    """
    size = len(samples)
    x = covariance[0][0]
    y = covariance[1][1]
    covariances = []
    if increasing:            
        for i in range(1, size+1, 2):                                                                        
            covariances.append([[x * i, 0], [0, y]]) 
        for i in range(size-1, -1, -2):
            covariances.append([[x * i, 0], [0, y]])
        covariances = np.array(covariances)
    else:
        covariances = np.array([covariance for _ in range(size)])
    return covariances

def plot_linear_data(samples, labels, x_range, y_range):
    """
        This function plots 2d linearly separable data
    """
    plt.scatter(samples[:, 0], samples[:, 1], c=labels)
    plt.xlim(0-(x_range * 1.1), x_range *  1.1 )
    plt.ylim(0-(y_range * 1.1), y_range * 1.1)
    plt.xticks(range(0, x_range+1, 5))
    plt.title('SVM Toy Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_samples_with_uncertainty(samples, labels, covariances, x_range, y_range):
    fig, ax = plt.subplots()
    print("Samples shape:", samples)
    print("Labels shape:", labels.shape)
    print("Covariances shape:", covariances.shape)
    ax.scatter(samples[:,0], samples[:,1], c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    set_of_labels = np.unique(labels) 
    colors = ['blue', 'red']
    for label, color in zip(set_of_labels, colors):
        indices = np.where(labels == label)
        class_samples = samples[indices]
        class_covariances = covariances[indices]

        ax.scatter(class_samples[:, 0], class_samples[:, 1], c=color, label=f'Class {label}', edgecolor='k')
        for mean, covariance in zip(class_samples, class_covariances):
            width, height = 4 * np.sqrt(covariance[0, 0]), 4 * np.sqrt(covariance[1, 1])
            ellipse = Ellipse(xy=mean, width=width, height=height, edgecolor=color, fc='None', lw=1)
            ax.add_patch(ellipse)

    ax.set_xlim(-0.5-x_range, x_range+0.5)
    ax.set_ylim(-0.5-y_range, y_range+0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Samples with Uncertainty and Decision Boundary')
    ax.legend()
    plt.show()


def make_meshgrid(h, x_range, y_range):
    x_min, x_max = 0-x_range, x_range
    y_min, y_max = 0-y_range, y_range
    xx, yy = np.meshgrid(np.arange(x_min-1, x_max+1, h), np.arange(y_min-1, y_max+1, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
    Z = np.sign(model.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, **params)
    return out


def plot_samples_with_uncertainty_and_boundary(model, samples, labels, covariances, x_range, y_range):
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots()

    #set up the grid to plot the predictions over
    xx, yy = make_meshgrid(0.01,x_range, y_range)

    plot_contours(ax, model, xx, yy)
    ax.scatter(samples[:,0], samples[:,1], c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    #plot samples and uncertainty ellipsoids
    set_of_labels = np.unique(labels)  #unique label values
    colors = ['blue', 'red']
    for label, color in zip(set_of_labels, colors):
        indices = np.where(labels == label)
        class_samples = samples[indices]
        class_covariances = covariances[indices]

        ax.scatter(class_samples[:, 0], class_samples[:, 1], c=color, label=f'Class {label}', edgecolor='k')
        for mean, covariance in zip(class_samples, class_covariances):
            width, height = 4 * np.sqrt(covariance[0, 0]), 4 * np.sqrt(covariance[1, 1])
            ellipse = Ellipse(xy=mean, width=width, height=height, edgecolor=color, fc='None', lw=1)
            ax.add_patch(ellipse)

    ax.set_xlim(-0.5-x_range, x_range+0.5)
    ax.set_ylim(-0.5-y_range, y_range+0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Samples with Uncertainty and Decision Boundary')
    ax.legend()
    plt.show()



