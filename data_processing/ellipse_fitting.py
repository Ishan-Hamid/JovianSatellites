import numpy as np

def ellipse_fit(x,y):
    # least squares fit of an ellipse using matrix eqns

    J = np.vstack([x**2, x*y, y**2, x, y]).T
    K = np.ones_like(x)
    JT = J.transpose()
    JTJ = np.dot(JT,J)
    invJTJ = np.linalg.inv(JTJ)
    vector = np.dot(invJTJ, np.dot(JT,K))

    return np.append(vector, -1)

def polyToParams(v,printMe):
   # B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   # center, axes, and tilt
   # v = np.array([A, B, C, D, E, F=-1]) (coefficients A..F) converts these to actual understandable values like centroid rotation axes length
   #Algebraic form: X.T * Amat * X --> polynomial form

   Amat = np.array(
   [
   [v[0],     v[1]/2.0, v[3]/2.0],
   [v[1]/2.0, v[2],     v[4]/2.0],
   [v[3]/2.0, v[4]/2.0, v[5]    ]
   ])

   if printMe: print('\nAlgebraic form of polynomial\n',Amat)

   A2=Amat[0:2,0:2]
   A2Inv=np.linalg.inv(A2)
   ofs=v[3:5]/2.0
   cc = -np.dot(A2Inv,ofs)
   if printMe: print('\nCenter at:',cc)

   # Center the ellipse at the origin
   Tofs=np.eye(3)
   Tofs[2,0:2]=cc
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   if printMe: print('\nAlgebraic form translated to center\n',R,'\n')

   R2=R[0:2,0:2]
   s1=-R[2, 2]
   RS=R2/s1
   (el,ec)=np.linalg.eig(RS)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   if printMe: print('\nAxes are\n',axes  ,'\n')

   rads=np.arctan2(ec[1,0],ec[0,0])
   deg=np.degrees(rads) #convert radians to degrees
   if printMe: print('Rotation is ',deg,'\n')

   inve=np.linalg.inv(ec) #inverse is actually the transpose here
   if printMe: print('\nRotation matrix\n',inve)

   # returns (center, axes, tilt degrees, rotation matrix)
   return (cc[0],cc[1],axes[0],axes[1],deg,inve)

def convert_to_physical(A, B, C, D, E, F):
    ## Takes A, B, C, D, E, F and converts them into ellipse physical params
    ## Basically my version of poltToParams
    ## Note: gives theta in radians
    ## Used in mcmc_finding_physical_params in MCMCFits

    x0 = (2*C*D - B*E)/(B**2-4*A*C)
    y0 = (2*A*E - B*D)/(B**2-4*A*C)
    a = -( np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C)+(np.sqrt((A-C)**2 + B**2)))) )/(B**2-4*A*C)
    b = -( np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C)-(np.sqrt((A-C)**2 + B**2)))) )/(B**2-4*A*C)
    theta = np.arctan2(-B, C-A)/2
    return x0, y0, a, b, theta

def model_ellipse_ex(x, y):
    #NOT THIS, NOT IN THE FORM WE NEED
    #used to be called 'model_ellipse'

    v = ellipse_fit(x, y)
    A, B, C, D, E, F = v[0], v[1], v[2], v[3], v[4], v[5]
    return A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F

def model_ellipse(x, y, *params):
    # used to be called 'model'

    return params[0] * x ** 2 + params[1] * x * y + params[2] * y ** 2 + params[3] * x + params[4] * y + params[5]

def physical_model(x, y, *params):
    x0, y0, a, b, theta = params
    return ((x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))**2/a**2 + (-(x-x0)*np.sin(theta) + (y-y0)*np.cos(theta))**2/b**2 - 1

