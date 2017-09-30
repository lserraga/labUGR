import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from .ltisys import lti, ZerosPolesGain, TransferFunction, StateSpace
from labugr.filters.filters import tf2zpk
    
__all__ = ['zplane']

def zplane(lti_system, ax=None):
    """
    Plot the complex z-plane given lti system represented by the zeros 
    poles gain, the transfer function or the state space.
    
    Parameters
    ----------
    z: lti object
        In any of the possible representation forms
    ax: matplotlib subplot
        Matplotlib subplot where the zplane will be plotted. If None, 
        the function creates one and shows it 

    """

    # Check if its a system object
    if not isinstance(lti_system, lti):
        raise Exception ("{} is not a lti object".format(lti_system))

    # Check if the system is in ZPK form
    if isinstance(lti_system, ZerosPolesGain):
        plot_zp(lti_system.zeros, lti_system.poles, ax)

    # If it's not in the zpk form, first convert it to zpk
    elif isinstance(lti_system, TransferFunction)or isinstance(lti_system, StateSpace):
        converted = lti_system.to_zpk()
        plot_zp(converted.zeros, converted.poles, ax)
    else:
        raise Exception ("{} does not have the correct format".format(lti_system))



def plot_zp(z, p, ax=None):
    """
    Plot the complex z-plane given zeros and poles.
    
    Parameters
    ----------
    z: array
        Array of zeros
    p: array
        Array of poles
    ax: matplotlib subplot
        If None, the function will show the plot 

    """
    show_plot = False

    if (ax==None):
        ax = plt.subplot(1, 1, 1)
        show_plot = True
    
    # Cirle, axes lines, grid, title and labels  
    unit_circle = patches.Circle((0,0), radius=1, fill=False,
                                 color='black', ls='solid', alpha=0.9)
    ax.add_patch(unit_circle)
    plt.axvline(0, color='black')
    plt.axhline(0, color='black')
    ax.grid(True)
    plt.title('Pole-zero plot')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    
    # Plot the poles
    poles = plt.plot(p.real, p.imag, 'x', color='red', markersize=9, alpha=0.5)
    
    # Plot the zeros
    zeros = plt.plot(z.real, z.imag,  'o', markersize=9, 
             color='none', alpha=0.5,
             markeredgecolor='blue'
             )

    # Scale axes to fit (in case zeros or poles greater than 1)
    r = 1.5 * np.amax(np.concatenate((abs(z), abs(p), [1])))
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])

    if show_plot:
        plt.show()



