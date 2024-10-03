'''
    Collocation points for the MFS
'''
import numpy as np


## 2D collocation points
def collocation_2d(n):
    '''
    Collocation points for the 2D MFS
    '''
    return np.array([(2*np.pi*i)/n for i in range(1,n+1)])


def source_points_2d(n: int) -> np.array:
    '''
    Source points for the 2D MFS
    '''
    arr = np.array([(2*np.pi*i)/n for i in range(1,n+1)])
    return np.array([arr, arr])


# ## 3D collocation points
# def collocation_3d(n):
#     '''
#     Collocation points for the 3D MFS
#     '''
#     return [(2*np.pi*i)/n for i in range(1,n+1)]


# def source_points_3d(n):
#     '''
#     Source points for the 3D MFS
#     '''
#     return [(2*np.pi*i)/n for i in range(1,n+1)]

