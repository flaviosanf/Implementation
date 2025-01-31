import numpy as np
import sys
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative

################
#### Constants
################
c = 2.99792458e08  # m/s
#LambdaCDM = 1.1056e-52  # 1/m^2
Mp = 2.18e-08  # kg
MSol = 1.989e30  # kg
RSol = 6.96e08  # m
hb = 1.054571817e-34  #m^2 kg/s
Gcte = 6.6743e-11  #Nm^2/kg^2
LambdaCDM = 1.54e-32

########################################
##### Differential Equation System #####
########################################


######### Non Integrate System #########
def systemaD(r, yv, arg):
    """ Sistema obtenido sin integrar """
    
    lamb, Lambda = arg
    y, y1, y2 = yv
    
    if abs(y) < 1e-10 and abs(r - 1) < 1e-10:
        f1 = y1
        f2 = y2
        f3 = -0.08333333333333333*(12*(3 - 4*y)*y1**2*lamb + 24*r*y1*(2*y1**2 + (-3 + 4*y)*y2)*lamb + r**3*y1*(5 + 12*y2**2*lamb)\
            - 3*r**2*(1-y + 4*y2*(4*y1**2 + y*y2)*lamb) + r**4*(y2 + 5*Lambda))/(r**2*(y1*(2 - 5*y + r*y1) + 3*r*y*y2)*lamb)
    else:
        f1 = y1
        f2 = y2  # y1'
        f3 = -0.08333333333333333*(r**4*y1 - 48*(-1+y)*y*y1*lamb - 48*r**2*y*y1*y2*lamb + 12*r*((-1+4*y)*y1**2 + 4*(-1+y)*y*y2)*lamb+\
            r**3*(-1+y+12*y*y2**2*lamb)+r**5*Lambda)/(r**2*y*(2-2*y+r*y1)*lamb)
    return [f1, f2, f3]

###Sistema Integrado##############
def systema(r, yv, arg):
    """ Sistema obtenido luego de integrar """
    
    lamb, Lambda = arg
    y, y1 = yv
    
    if y == 0.0: # usar condicion del horizonte
        f1 = df1Rheq(r, arg)
        f2 = 0  # suposición fuerte
    else:
        f1 = y1
        f2 = (-(r**2*(3 - 3*r + Lambda*r**3 + 3*r*y)) - 72*lamb*(-1 + y)*y*y1 + 36*lamb*r*y1**2 + 12*lamb*r**2*y1**3)/(36.*lamb*r*y*(2 - 2*y + r*y1))    
    return [f1, f2]

############## system for f'in rh ################

def df1Rheq(r, arg):
    """ Sistema de f' en rh """
    lamb, Lambda = arg
    
    # rama de soluciones para f'(rh)
    df1 = (-12/r - (24*3**0.3333333333333333*lamb)/(r**2*((24*lamb**3)/r**3 - lamb**2*(3 - 3*r + Lambda*r**3)\
           + np.sqrt((lamb**4*(3 - 3*r + Lambda*r**3)*(-48*lamb + 3*r**3 - 3*r**4 + Lambda*r**6))/r**3))**0.3333333333333333)\
           - (2*3**0.6666666666666666*((24*lamb**3)/r**3 - lamb**2*(3 - 3*r + Lambda*r**3) + np.sqrt((lamb**4*(3 - 3*r\
           + Lambda*r**3)*(-48*lamb + 3*r**3 - 3*r**4 + Lambda*r**6))/r**3))**0.3333333333333333)/lamb)/12.
    df2 = -(1/r) + (3**0.3333333333333333*(1 + (0,1)*np.sqrt(3))*lamb)/(r**2*((24*lamb**3)/r**3 - lamb**2*(3 - 3*r + Lambda*r**3)\
          + np.sqrt((lamb**4*(3 - 3*r + Lambda*r**3)*(-48*lamb + 3*r**3 - 3*r**4 + Lambda*r**6))/r**3))**0.3333333333333333) + ((1\
          - (0,1)*np.sqrt(3))*((24*lamb**3)/r**3 - lamb**2*(3 - 3*r + Lambda*r**3) +  np.sqrt((lamb**4*(3 - 3*r + Lambda*r**3)*(-48*lamb\
          + 3*r**3 - 3*r**4 + Lambda*r**6))/r**3))**0.3333333333333333)/(4.*3**0.3333333333333333*lamb)
    df3 = -(1/r) + (3**0.3333333333333333*(1 - (0,1)*np.sqrt(3))*lamb)/(r**2*((24*lamb**3)/r**3 - lamb**2*(3 - 3*r + Lambda*r**3) + np.sqrt((lamb**4*(3\
          - 3*r + Lambda*r**3)*(-48*lamb + 3*r**3 - 3*r**4 + Lambda*r**6))/r**3))**0.3333333333333333) + ((1 + (0,1)* np.sqrt(3))*((24*lamb**3)/r**3\
          - lamb**2*(3 - 3*r + Lambda*r**3) + np.sqrt((lamb**4*(3 - 3*r + Lambda*r**3)*(-48*lamb + 3*r**3 - 3*r**4\
          + Lambda*r**6))/r**3))**0.3333333333333333)/(4.*3**0.3333333333333333*lamb)
          
    # Selecting Real Branches
    sols = np.array([df1, df2, df3])
    print(sols)
    ind = np.imag(sols) == 0.0
    solf = np.real(sols[ind])
    if len(solf)>1:  # escogiendo solo una solución, por default la 0
        solf = np.real(solf[0])
    elif len(solf)==0:
        sys.exit("No hay soluciones reales para la derivada") 
    
    return solf



################ Resolution Differential Equation System ########################

def NumSol(C0, lamB, Npt=50000, iter='d',
           xmin=0, xmax=10, metodo='RK45', Rtol=1e-09, Atol=1e-10, info=False):
    """
    C0 -> In Sun Masses 
    """
    
    # Reesaling Cosmological Constant
    rs = 2*(C0*MSol)*Gcte/c**2  # se podia haber usado que G = hb*c/Mp**2
    LambdaB = LambdaCDM*rs**2
    
    sol, Event = solut(lamB, LambdaB, yBound=0, iter=iter,
          xmin=xmin, xmax=xmax, metodo=metodo, Rtol=Rtol, Atol=Atol, Npt=Npt)
    ######### Lets make this solution continous function ############
    solCont = interp1d(sol.t,sol.y[0], kind = 'linear', fill_value='extrapolate')


    # info
    if info:
        #lamF = 16*lamB*(C0*MSol)**4/
        print('Sch. radio -> ', rs)
        if Event[0]==0:
            print('Horizon located in r =', Event[0], 'Naked Singularity')
        else:
            print('Horizon located in r = ', Event[0])
    return sol, solCont

def solut(lamB, LambdaB, iter='d', yBound=0,
          xmin=0, xmax=10, metodo='RK45', Rtol = 1e-09, Atol = 1e-10, Npt=50000):  #Modificacion
    """
    yBound = [SAsy, SeW, fH]
    
    """
    yBoundOp = [SAsy, SeW, fNH]

    rB = xmin if iter=='l' else xmax
    fb, dfb = yBoundOp[yBound](rB, lamB, LambdaB)  # Boundaries conditions
    
    arg = [lamB, LambdaB]
    yBounds = [fb[0], dfb[0]]
    x_span = np.linspace(xmax, xmin, Npt)
    
    def Crh(r, U, arg): return U[0] #-parm
    Crh.direction = -1
    Crh.terminal = True
    print(yBounds)
    temp = solve_ivp(systema, [xmax, xmin], yBounds, t_eval=x_span, args=(arg,), method=metodo, rtol=Rtol, atol=Atol, events=(Crh))
    if temp.t_events[0].size > 0 and temp.y_events[0].size > 0:
      Event= temp.t_events[0] ##Where the horizon is located
    else:
        Event = [0]


    return temp, Event

###############################################
########## Series Aproximation ################

########## Weak Coupling Series ###############
def SeW(r, lamB, LambdaB):

    lamb, Lambda = lamB, LambdaB
    ep = lamb*Lambda**2 #epsilon

    f = (-504*ep*(6561 + 32*ep*(-3645 + 32*ep*(1701 + 880*ep*(-27 + 364*ep))))*Lambda**9*r**27\
        + (-19683 + 16*ep*(2187 + 16*ep*(-729 + 64*ep*(81 - 660*ep + 5824*ep**2))))*Lambda**11*r**33\
        + 3*Lambda**10*r**30*(-19683 + 16*ep*(6561 + 16*ep*(-3645 + 64*ep*(567 + 44*ep*(-135 + 1456*ep)))) + 19683*r)\
        - 7776*ep**2*Lambda**7*r**21*(1171017 + 32*ep*(5861349 + 16*ep*(-17148249 + 482819200*ep)) + 4698*(-243 + 256*ep*(27\
        + ep*(-549 + 9664*ep)))*r) + 324*ep*Lambda**8*r**24*(-16767 + 16*ep*(-278721 + 16*ep*(545103 + 32*ep*(-364101 + 6625040*ep)))\
        + 81*(243 + 16*ep*(-243 + 128*ep*(27 - 366*ep + 4832*ep**2)))*r) - 559872*ep**3*Lambda**5*r**15*(147605301 + 160*ep*(780100653\
        + 70231493552*ep) + 324*(-820665 + 16*ep*(-17996601 + 1016718160*ep))*r + 4372785*(27 + 32*ep*(-33 + 848*ep))*r**2)\
        + 11664*ep**2*Lambda**6*r**18*(-1134567 + 16*ep*(-114338709 + 16*ep*(-835687197 + 54672001888*ep)) + 39366*(57 + 32*ep*(1233\
        + 104*ep*(-495 + 13216*ep)))*r + 4536*(-243 + 80*ep*(81 + 16*ep*(-99 + 1696*ep)))*r**2) - 10077696*ep**4*Lambda**3*r**9*(145987689897\
        + 361846258043728*ep + 324*r*(-3*(389713185 + 577493868544*ep) + 8*r*(124810974 + 79717337458*ep + 3888999*(-9 + 448*ep)*r)))\
        + 419904*ep**3*Lambda**4*r**12*(27*(-8745152 + 81*r*(298435 + 96*r*(-2839 + 855*r))) - 144*ep*(8442168853 + 81*r*(-173568119\
        + 84*r*(835327 + 6840*r))) + 256*ep**2*(-2865473751949 + 324*r*(5821102400 + 3*r*(172024277 + 670320*r))))\
        - 120932352*ep**5*Lambda*r**3*(352754921573507 + 972*r*(-1224212978588 + 3*r*(510400276841 + 144*r*(-1945000180 + 394282701*r))))\
        + 5038848*ep**4*Lambda**2*r**6*(-295094506413 - 3660535297585808*ep + 486*r*(2149679163 + 18474528584768*ep + 6*r*(-471430266\
        - 2473768359088*ep + 27*r*(10106133 + 23908762768*ep + 2167296*(-1 + 48*ep)*r)))) + 181398528*ep**5*(-204713167318553\
        + 243*r*(3630718747941 + 32*r*(-193932488245 + 27*r*(6079956100 + 27*r*(-94276103 + 15592962*r))))))/(59049.*Lambda**10*r**31)

    df = (2016*ep*(6561 + 32*ep*(-3645 + 32*ep*(1701 + 880*ep*(-27 + 364*ep))))*Lambda**9*r**27 + 3*(19683 - 16*ep*(6561 + 16*ep*(-3645\
        + 64*ep*(567 + 44*ep*(-135 + 1456*ep)))))*Lambda**10*r**30 + 2*(-19683 + 16*ep*(2187 + 16*ep*(-729 + 64*ep*(81 - 660*ep + 5824*ep**2))))*Lambda**11*r**33\
        + 15552*ep**2*Lambda**7*r**21*(5855085 + 160*ep*(5861349 + 16*ep*(-17148249 + 482819200*ep)) + 21141*(-243 + 256*ep*(27 + ep*(-549 + 9664*ep)))*r)\
        - 324*ep*Lambda**8*r**24*(-1701*(69 + 18352*ep) + 1792*ep**2*(545103 + 32*ep*(-364101 + 6625040*ep)) + 486*(243 + 16*ep*(-243 + 128*ep*(27 - 366*ep\
        + 4832*ep**2)))*r) + 1119744*ep**3*Lambda**5*r**15*(1180842408 + 1280*ep*(780100653 + 70231493552*ep) + 2430*(-820665 + 16*ep*(-17996601 + 1016718160*ep))*r\
        + 30609495*(27 + 32*ep*(-33 + 848*ep))*r**2) - 11664*ep**2*Lambda**6*r**18*(-14749371 + 208*ep*(-114338709 + 16*ep*(-835687197 + 54672001888*ep))\
        + 472392*(57 + 32*ep*(1233 + 104*ep*(-495 + 13216*ep)))*r + 49896*(-243 + 80*ep*(81 + 16*ep*(-99 + 1696*ep)))*r**2) - 839808*ep**3*Lambda**4*r**12*(-152*(14757444\
        + ep*(75979519677 + 45847580031184*ep)) + 729*(8057745 + 16*ep*(1562113071 + 372550553600*ep))*r + 66096*(-76653 + 2*ep*(-52625601 + 2752388432*ep))*r**2\
        + 53187840*(27 + 112*ep*(-9 + 224*ep))*r**3) + 20155392*ep**4*Lambda**3*r**9*(1605864588867 + 3980308838481008*ep + 162*r*(-63*(389713185 + 577493868544*ep)\
        + 8*r*(40*(62405487 + 39858668729*ep) + 73890981*(-9 + 448*ep)*r))) + 483729408*ep**5*Lambda*r**3*(2469284451014549 + 1458*r*(-5508958403646 + r*(6635203598933\
        + 288*r*(-12156251125 + 2365696206*r)))) - 5038848*ep**4*Lambda**2*r**6*(-25*(295094506413 + 3660535297585808*ep) + 5832*r*(4299358326 + 36949057169536*ep\
        + r*(-23*(235715133 + 1236884179544*ep) + 27*r*(111167463 + 262996390448*ep + 22756608*(-1 + 48*ep)*r)))) - 181398528*ep**5*(-6346108186875143\
        + 486*r*(54460781219115 + 16*r*(-5624042159105 + 27*r*(170238770800 + 81*r*(-848484927 + 135139004*r))))))/(59049.*Lambda**10*r**32)

    return f, df

###### Asymptotical Series #############################

def SAsy(r, lamB, LambdaB):
    """
    Serie asintótica
    """
    lamb, Lambda = lamB, LambdaB
    LamEff = LamEffsol(lamb, Lambda)
    
    f = 1 - (32*lamb**2*LamEff*(4819 + (2881024*lamb*LamEff**2)/3.))/((1 + (16*lamb*LamEff**2)/3.)**7*r**10) + (150336*lamb**2*LamEff)/((1\
        + (16*lamb*LamEff**2)/3.)**5*r**9) - (4*lamb*(23 + (20192*lamb*LamEff**2)/3.))/((1 + (16*lamb*LamEff**2)/3.)**5*r**7) + (108*lamb)/((1\
        + (16*lamb*LamEff**2)/3.)**3*r**6) - (56*lamb*LamEff)/((1 + (16*lamb*LamEff**2)/3.)**3*r**4) - 1/((1 + (16*lamb*LamEff**2)/3.)*r) - (LamEff*r**2)/3.
    
    df = ((699840*lamb**2*LamEff*(14457 + 2881024*lamb*LamEff**2))/(3 + 16*lamb*LamEff**2)**7 - (986354496*lamb**2*LamEff*r)/(3 + 16*lamb*LamEff**2)**5\
         + (6804*lamb*(69 + 20192*lamb*LamEff**2)*r**3)/(3 + 16*lamb*LamEff**2)**5 - (52488*lamb*r**4)/(3 + 16*lamb*LamEff**2)**3 + (18144*lamb*LamEff*r**6)/(3\
         + 16*lamb*LamEff**2)**3 + (9*r**9)/(3 + 16*lamb*LamEff**2) - 2*LamEff*r**12)/(3.*r**11)

    return f, df

####### Efective Cosmological Constant ####################
def LamEffsol(lamb, Lambda, check=True):
    
    lamb = lamb + 0j if lamb<0 else lamb  # para prevenir error en sqrt
    
    Leff1 = (-3**0.6666666666666666 + 3**0.3333333333333333*lamb*((6*np.sqrt(lamb)*Lambda + np.sqrt(3\
        + 36*lamb*Lambda**2))/lamb**1.5)**0.6666666666666666)/(4.*lamb*((6*np.sqrt(lamb)*Lambda + np.sqrt(3 + 36*lamb*Lambda**2))/lamb**1.5)**0.3333333333333333)
    
    Leff2 = ((-3)**0.3333333333333333*(3**0.3333333333333333 + (-1)**0.3333333333333333*lamb*((6*np.sqrt(lamb)*Lambda\
        + np.sqrt(3 + 36*lamb*Lambda**2))/lamb**1.5)**0.6666666666666666))/(4.*lamb*((6*np.sqrt(lamb)*Lambda + np.sqrt(3 + 36*lamb*Lambda**2))/lamb**1.5)**0.3333333333333333)
    
    Leff3 = -0.25*((-3)**0.3333333333333333*((-3)**0.3333333333333333 + lamb*((6*np.sqrt(lamb)*Lambda + np.sqrt(3 +\
        36*lamb*Lambda**2))/lamb**1.5)**0.6666666666666666))/(lamb*((6*np.sqrt(lamb)*Lambda + np.sqrt(3 + 36*lamb*Lambda**2))/lamb**1.5)**0.3333333333333333)
    
    # It selects the real branch with no ghosts
    solP = np.array([Leff1, Leff2, Leff3])
    
   # ind1 = np.imag(solP) == 0.0; ind2 = np.real(solP) > 0.0
   # ind = ind1*ind2
   # solf = np.real(solP[ind])
    
    ind1 = np.imag(solP) == 0.0 ###It  chooses the real branch, but it might have ghosts 
    ind = ind1
    solf = np.real(solP[ind])
    
    if check:
        print('lambda ->',lamb, ' Lambda -> ', Lambda)
        print('numerica -> ', solP)
        if np.real(lamb)>0 or np.real(lamb)<-1/(12*Lambda**2):
            # print('caso 1', -1/(12*Lambda), np.real(lamb))
            LeffR = np.cbrt((9*Lambda)/(32.*np.real(lamb)) - np.sqrt(27/(4096.*np.real(lamb)**3) + (81*Lambda**2)/(256.*np.real(lamb)**2)))\
                + np.cbrt((9*Lambda)/(32.*np.real(lamb)) + np.sqrt(27/(4096.*np.real(lamb)**3) + (81*Lambda**2)/(256.*np.real(lamb)**2)))
            print('analitico -> ', LeffR)
        elif np.real(lamb)>-1/(12*Lambda**2) and np.real(lamb)<0:
            #print('caso 2', -1/(12*Lambda), np.real(lamb))
            theta = np.arccos(((9*Lambda)/(32.*np.real(lamb)))/np.sqrt((3/(16*abs(lamb)))**3))
            LeffR = [np.sqrt(3/(4*abs(lamb))*np.cos((theta+2*k*np.pi)/3)) for k in range(3)]
            print('analitico -> ', LeffR)
        else:
            print('analitico -> ', [3*Lambda/2, -3*Lambda])
        
    
    if len(solf)>1:  # escogiendo solo una solución, por default la 0
        solf = np.real(solf[0])
    elif len(solf)==0:
       sys.exit("No hay soluciones reales y positivas") 
    
    return solf

################ Near Horizon Ansatz #####################################################
def fNH(r, lamB, LambdaB, rh, a2):
    f = a1(rh, lamB, LambdaB)*(r-rh) + a2*(r-rh)**2+a3(rh,a1,a2,lamB,LambdaB)*(r-rh)**3+a4(rh,a1,a2,lamB,LambdaB)*(r-rh)**4+\
        a5(rh,a1,a2,lamB,LambdaB)*(r-rh)**5+a6(rh,a1,a2,lamB,LambdaB)*(r-rh)**6+0*a7(rh,a1,a2,lamB,LambdaB)*(r-rh)**7
    df = a1(rh,lamB, LambdaB) + 2*a2*(r-rh) + 3*a3(rh,a1,a2,lamB,LambdaB)*(r-rh)**2+4*a4(rh,a1,a2,lamB,LambdaB)**3\
        +5*a5(rh,a1,a2,lamB,LambdaB)*(r-rh)**4 + 6*a6(rh,a1,a2,lamB,LambdaB)*(r-rh)**5+0*7*a7(rh,a1,a2,lamB,LambdaB)**6
    return f,df

def a1(rh, lambdaB, Lambda):
    return (rh**3 - rh*np.sqrt(-48*lambdaB + 48*Lambda*lambdaB*rh**2 + rh**4))/(24.*lambdaB)

def a3(rh, a1Func, a2, lambdaB, Lambda):
  a1Val = a1Func(rh, lambdaB, Lambda)
  a3Val = -0.027777777777777776*(36*a1Val**2*lambdaB + 24*a1Val**3*lambdaB*rh -\
          72*a1Val*a2*lambdaB*rh - 48*a1Val**2*a2*lambdaB*rh**2 + a1Val*rh**3 +\
          24*a1Val*a2**2*lambdaB*rh**3 + a2*rh**4 + Lambda*rh**4)/\
         (a1Val*lambdaB*rh**2*(2 + a1Val*rh))
  return a3Val

def a4(rh,a1Func,a2,lambdaB, Lambda):
    a1Val = a1Func(rh,lambdaB,Lambda)
    a4Val = -0.00028935185185185184*(1728*a1Val**3*lambdaB**2 + 3600*a1Val**4*lambdaB**2*rh -\
          3456*a1Val**2*a2*lambdaB**2*rh + 1728*a1Val**5*lambdaB**2*rh**2 -\
          10080*a1Val**3*a2*lambdaB**2*rh**2 + 132*a1Val**2*lambdaB*rh**3 -\
          5760*a1Val**4*a2*lambdaB**2*rh**3 + 9792*a1Val**2*a2**2*lambdaB**2*rh**3 +\
          120*a1Val**3*lambdaB*rh**4 + 264*a1Val*a2*lambdaB*rh**4 +\
          192*a1Val*Lambda*lambdaB*rh**4 + 8064*a1Val**3*a2**2*lambdaB**2*rh**4 +\
          60*a1Val**2*a2*lambdaB*rh**5 - 48*a2**2*lambdaB*rh**5 +\
          156*a1Val**2*Lambda*lambdaB*rh**5 - 48*a2*Lambda*lambdaB*rh**5 -\
          3456*a1Val**2*a2**3*lambdaB**2*rh**5 - a1Val*rh**6 - 192*a1Val*a2**2*lambdaB*rh**6 -\
          168*a1Val*a2*Lambda*lambdaB*rh**6 - a2*rh**7 - Lambda*rh**7)/\
         (a1Val**2*lambdaB**2*rh**3*(2 + a1Val*rh)**2)
    return a4Val

def a5(rh,a1Fun,a2,lambdaB, Lambda):
    a1Val = a1Fun(rh,lambdaB, Lambda)
    a5Val = -1.6075102880658436e-6*(611712*a1Val**5*lambdaB**3 + 1009152*a1Val**6*lambdaB**3*rh -\
          2820096*a1Val**4*a2*lambdaB**3*rh - 11232*a1Val**3*lambdaB**2*rh**2 +\
          414720*a1Val**7*lambdaB**3*rh**2 - 4890240*a1Val**5*a2*lambdaB**3*rh**2 +\
          3317760*a1Val**3*a2**2*lambdaB**3*rh**2 + 26928*a1Val**4*lambdaB**2*rh**3 +\
          55296*a1Val**2*a2*lambdaB**2*rh**3 + 16128*a1Val**2*Lambda*lambdaB**2*rh**3 -\
          2115072*a1Val**6*a2*lambdaB**3*rh**3 + 7831296*a1Val**4*a2**2*lambdaB**3*rh**3 +\
          27648*a1Val**2*a2**3*lambdaB**3*rh**3 + 28224*a1Val**5*lambdaB**2*rh**4 +\
          30816*a1Val**3*a2*lambdaB**2*rh**4 - 34560*a1Val*a2**2*lambdaB**2*rh**4 +\
          67104*a1Val**3*Lambda*lambdaB**2*rh**4 - 24192*a1Val*a2*Lambda*lambdaB**2*rh**4 +\
          4147200*a1Val**5*a2**2*lambdaB**3*rh**4 - 4216320*a1Val**3*a2**3*lambdaB**3*rh**4 -\
          540*a1Val**2*lambdaB*rh**5 - 27072*a1Val**4*a2*lambdaB**2*rh**5 -\
          198720*a1Val**2*a2**2*lambdaB**2*rh**5 + 6912*a2**3*lambdaB**2*rh**5 +\
          42624*a1Val**4*Lambda*lambdaB**2*rh**5 - 146304*a1Val**2*a2*Lambda*lambdaB**2*rh**5 +\
          6912*a2**2*Lambda*lambdaB**2*rh**5 - 3732480*a1Val**4*a2**3*lambdaB**3*rh**5 -\
          24*a1Val**3*lambdaB*rh**6 - 432*a1Val*a2*lambdaB*rh**6 - 504*a1Val*Lambda*lambdaB*rh**6 -\
          84960*a1Val**3*a2**2*lambdaB**2*rh**6 + 26496*a1Val*a2**3*lambdaB**2*rh**6 -\
          113760*a1Val**3*a2*Lambda*lambdaB**2*rh**6 + 23040*a1Val*a2**2*Lambda*lambdaB**2*rh**6 +\
          1244160*a1Val**3*a2**4*lambdaB**3*rh**6 + 876*a1Val**2*a2*lambdaB*rh**7 +\
          288*a2**2*lambdaB*rh**7 + 420*a1Val**2*Lambda*lambdaB*rh**7 +\
          384*a2*Lambda*lambdaB*rh**7 + 96*Lambda**2*lambdaB*rh**7 +\
          84096*a1Val**2*a2**3*lambdaB**2*rh**7 + 72000*a1Val**2*a2**2*Lambda*lambdaB**2*rh**7 +\
          a1Val*rh**8 + 1032*a1Val*a2**2*lambdaB*rh**8 + 1488*a1Val*a2*Lambda*lambdaB*rh**8 +\
          480*a1Val*Lambda**2*lambdaB*rh**8 + a2*rh**9 + Lambda*rh**9)/\
         (a1Val**3*lambdaB**3*rh**3*(2 + a1Val*rh)**3)
    return a5Val

def a6(rh, a1Fun, a2,lambdaB, Lambda):
    a1Val = a1Fun(rh,lambdaB, Lambda)
    a6Val = -5.581632944673068e-9*(229920768*a1Val**6*lambdaB**4 + 708300288*a1Val**7*lambdaB**4*rh -\
          970444800*a1Val**5*a2*lambdaB**4*rh - 4727808*a1Val**4*lambdaB**3*rh**2 +\
          693411840*a1Val**8*lambdaB**4*rh**2 - 3589899264*a1Val**6*a2*lambdaB**4*rh**2 +\
          991346688*a1Val**4*a2**2*lambdaB**4*rh**2 + 8294400*a1Val**5*lambdaB**3*rh**3 +\
          16174080*a1Val**3*a2*lambdaB**3*rh**3 + 1907712*a1Val**3*Lambda*lambdaB**3*rh**3 +\
          218972160*a1Val**9*lambdaB**4*rh**3 - 3928559616*a1Val**7*a2*lambdaB**4*rh**3 +\
          6249166848*a1Val**5*a2**2*lambdaB**4*rh**3 + 43794432*a1Val**3*a2**3*lambdaB**4*rh**3 +\
          32906304*a1Val**6*lambdaB**3*rh**4 + 42384384*a1Val**4*a2*lambdaB**3*rh**4 -\
          20901888*a1Val**2*a2**2*lambdaB**3*rh**4 + 33267456*a1Val**4*Lambda*lambdaB**3*rh**4 -\
          8128512*a1Val**2*a2*Lambda*lambdaB**3*rh**4 - 1343692800*a1Val**8*a2*lambdaB**4*rh**4 +\
          8683739136*a1Val**6*a2**2*lambdaB**4*rh**4 - 3904339968*a1Val**4*a2**3*lambdaB**4*rh**4 -\
          7962624*a1Val**2*a2**4*lambdaB**4*rh**4 - 187488*a1Val**3*lambdaB**2*rh**5 +\
          18420480*a1Val**7*lambdaB**3*rh**5 - 21848832*a1Val**5*a2*lambdaB**3*rh**5 -\
          160330752*a1Val**3*a2**2*lambdaB**3*rh**5 + 10948608*a1Val*a2**3*lambdaB**3*rh**5 +\
          58150656*a1Val**5*Lambda*lambdaB**3*rh**5 -\
          87782400*a1Val**3*a2*Lambda*lambdaB**3*rh**5 +\
          7962624*a1Val*a2**2*Lambda*lambdaB**3*rh**5 +\
          3459760128*a1Val**7*a2**2*lambdaB**4*rh**5 - 8846475264*a1Val**5*a2**3*lambdaB**4*rh**5 -\
          37158912*a1Val**3*a2**4*lambdaB**4*rh**5 - 80064*a1Val**4*lambdaB**2*rh**6 -\
          57024*a1Val**2*a2*lambdaB**2*rh**6 - 183744*a1Val**2*Lambda*lambdaB**2*rh**6 -\
          38029824*a1Val**6*a2*lambdaB**3*rh**6 - 170221824*a1Val**4*a2**2*lambdaB**3*rh**6 +\
          54798336*a1Val**2*a2**3*lambdaB**3*rh**6 - 1990656*a2**4*lambdaB**3*rh**6 +\
          26058240*a1Val**6*Lambda*lambdaB**3*rh**6 -\
          186997248*a1Val**4*a2*Lambda*lambdaB**3*rh**6 +\
          37739520*a1Val**2*a2**2*Lambda*lambdaB**3*rh**6 - 1990656*a2**3*Lambda*lambdaB**3*rh**6 -\
          4598415360*a1Val**6*a2**3*lambdaB**4*rh**6 + 3397386240*a1Val**4*a2**4*lambdaB**4*rh**6 +\
          189216*a1Val**5*lambdaB**2*rh**7 + 1514880*a1Val**3*a2*lambdaB**2*rh**7 +\
          321408*a1Val*a2**2*lambdaB**2*rh**7 + 503136*a1Val**3*Lambda*lambdaB**2*rh**7 +\
          414720*a1Val*a2*Lambda*lambdaB**2*rh**7 + 82944*a1Val*Lambda**2*lambdaB**2*rh**7 -\
          25754112*a1Val**5*a2**2*lambdaB**3*rh**7 + 221681664*a1Val**3*a2**3*lambdaB**3*rh**7 -\
          8626176*a1Val*a2**4*lambdaB**3*rh**7 - 98620416*a1Val**5*a2*Lambda*lambdaB**3*rh**7 +\
          168086016*a1Val**3*a2**2*Lambda*lambdaB**3*rh**7 -\
          7630848*a1Val*a2**3*Lambda*lambdaB**3*rh**7 +\
          3105423360*a1Val**5*a2**4*lambdaB**4*rh**7 + 684*a1Val**2*lambdaB*rh**8 +\
          896688*a1Val**4*a2*lambdaB**2*rh**8 + 1686528*a1Val**2*a2**2*lambdaB**2*rh**8 -\
          124416*a2**3*lambdaB**2*rh**8 + 748656*a1Val**4*Lambda*lambdaB**2*rh**8 +\
          2431872*a1Val**2*a2*Lambda*lambdaB**2*rh**8 - 186624*a2**2*Lambda*lambdaB**2*rh**8 +\
          694080*a1Val**2*Lambda**2*lambdaB**2*rh**8 - 62208*a2*Lambda**2*lambdaB**2*rh**8 +\
          113909760*a1Val**4*a2**3*lambdaB**3*rh**8 - 21676032*a1Val**2*a2**4*lambdaB**3*rh**8 +\
          130429440*a1Val**4*a2**2*Lambda*lambdaB**3*rh**8 -\
          17694720*a1Val**2*a2**3*Lambda*lambdaB**3*rh**8 -\
          836075520*a1Val**4*a2**5*lambdaB**4*rh**8 - 2556*a1Val**3*lambdaB*rh**9 -\
          432*a1Val*a2*lambdaB*rh**9 - 72*a1Val*Lambda*lambdaB*rh**9 -\
          290304*a1Val**3*a2**2*lambdaB**2*rh**9 - 520704*a1Val*a2**3*lambdaB**2*rh**9 +\
          717120*a1Val**3*a2*Lambda*lambdaB**2*rh**9 - 777600*a1Val*a2**2*Lambda*lambdaB**2*rh**9 +\
          597600*a1Val**3*Lambda**2*lambdaB**2*rh**9 - 267264*a1Val*a2*Lambda**2*lambdaB**2*rh**9 -\
          66576384*a1Val**3*a2**4*lambdaB**3*rh**9 -\
          56954880*a1Val**3*a2**3*Lambda*lambdaB**3*rh**9 - 7320*a1Val**2*a2*lambdaB*rh**10 -\
          1296*a2**2*lambdaB*rh**10 - 6240*a1Val**2*Lambda*lambdaB*rh**10 -\
          2112*a2*Lambda*lambdaB*rh**10 - 816*Lambda**2*lambdaB*rh**10 -\
          1281024*a1Val**2*a2**3*lambdaB**2*rh**10 -\
          1994112*a1Val**2*a2**2*Lambda*lambdaB**2*rh**10 -\
          740160*a1Val**2*a2*Lambda**2*lambdaB**2*rh**10 - a1Val*rh**11 -\
          4896*a1Val*a2**2*lambdaB*rh**11 - 8592*a1Val*a2*Lambda*lambdaB*rh**11 -\
          3720*a1Val*Lambda**2*lambdaB*rh**11 - a2*rh**12 - Lambda*rh**12)/\
          (a1Val**4*lambdaB**4*rh**4*(2 + a1Val*rh)**4)
    return a6Val

def a7(rh, a1Fun,a2, lambdaB, Lambda):
    a1Val = a1Fun(rh, lambdaB, Lambda)
    a7Val = -1.328960224922159e-11*(61989027840*a1Val**7*lambdaB**5 +\
          444914601984*a1Val**8*lambdaB**5*rh - 237564887040*a1Val**6*a2*lambdaB**5*rh -\
          1224253440*a1Val**5*lambdaB**4*rh**2 + 870954799104*a1Val**9*lambdaB**5*rh**2 -\
          2474060931072*a1Val**7*a2*lambdaB**5*rh**2 +\
          206391214080*a1Val**5*a2**2*lambdaB**5*rh**2 - 1379524608*a1Val**6*lambdaB**4*rh**3 +\
          7494819840*a1Val**4*a2*lambdaB**4*rh**3 + 806215680*a1Val**4*Lambda*lambdaB**4*rh**3 +\
          666272563200*a1Val**10*lambdaB**5*rh**3 - 5552715939840*a1Val**8*a2*lambdaB**5*rh**3 +\
          4796386099200*a1Val**6*a2**2*lambdaB**5*rh**3 +\
          55897620480*a1Val**4*a2**3*lambdaB**5*rh**3 + 24693838848*a1Val**7*lambdaB**4*rh**4 +\
          51829714944*a1Val**5*a2*lambdaB**4*rh**4 - 15288238080*a1Val**3*a2**2*lambdaB**4*rh**4 +\
          16614512640*a1Val**5*Lambda*lambdaB**4*rh**4 -\
          3841966080*a1Val**3*a2*Lambda*lambdaB**4*rh**4 +\
          177964646400*a1Val**11*lambdaB**5*rh**4 - 4630910828544*a1Val**9*a2*lambdaB**5*rh**4 +\
          13680010985472*a1Val**7*a2**2*lambdaB**5*rh**4 -\
          3193378504704*a1Val**5*a2**3*lambdaB**5*rh**4 -\
          23887872000*a1Val**3*a2**4*lambdaB**5*rh**4 - 72596736*a1Val**4*lambdaB**3*rh**5 +\
          42073426944*a1Val**8*lambdaB**4*rh**5 + 13675889664*a1Val**6*a2*lambdaB**4*rh**5 -\
          134837084160*a1Val**4*a2**2*lambdaB**4*rh**5 +\
          13974405120*a1Val**2*a2**3*lambdaB**4*rh**5 +\
          61322655744*a1Val**6*Lambda*lambdaB**4*rh**5 -\
          50063671296*a1Val**4*a2*Lambda*lambdaB**4*rh**5 +\
          6768230400*a1Val**2*a2**2*Lambda*lambdaB**4*rh**5 -\
          1320521564160*a1Val**10*a2*lambdaB**5*rh**5 +\
          13259799429120*a1Val**8*a2**2*lambdaB**5*rh**5 -\
          15298669117440*a1Val**6*a2**3*lambdaB**5*rh**5 -\
          139345920000*a1Val**4*a2**4*lambdaB**5*rh**5 +\
          3822059520*a1Val**2*a2**5*lambdaB**5*rh**5 - 323585280*a1Val**5*lambdaB**3*rh**6 +\
          52627968*a1Val**3*a2*lambdaB**3*rh**6 - 93934080*a1Val**3*Lambda*lambdaB**3*rh**6 +\
          17609011200*a1Val**9*lambdaB**4*rh**6 - 92397127680*a1Val**7*a2*lambdaB**4*rh**6 -\
          291067748352*a1Val**5*a2**2*lambdaB**4*rh**6 +\
          87007592448*a1Val**3*a2**3*lambdaB**4*rh**6 - 5971968000*a1Val*a2**4*lambdaB**4*rh**6 +\
          69619046400*a1Val**7*Lambda*lambdaB**4*rh**6 -\
          232624410624*a1Val**5*a2*Lambda*lambdaB**4*rh**6 +\
          43639824384*a1Val**3*a2**2*Lambda*lambdaB**4*rh**6 -\
          4538695680*a1Val*a2**3*Lambda*lambdaB**4*rh**6 +\
          4225764556800*a1Val**9*a2**2*lambdaB**5*rh**6 -\
          19400057487360*a1Val**7*a2**3*lambdaB**5*rh**6 +\
          6524016721920*a1Val**5*a2**4*lambdaB**5*rh**6 +\
          20384317440*a1Val**3*a2**5*lambdaB**5*rh**6 + 139665600*a1Val**6*lambdaB**3*rh**7 +\
          1566729216*a1Val**4*a2*lambdaB**3*rh**7 + 240122880*a1Val**2*a2**2*lambdaB**3*rh**7 +\
          93699072*a1Val**4*Lambda*lambdaB**3*rh**7 +\
          343719936*a1Val**2*a2*Lambda*lambdaB**3*rh**7 +\
          47333376*a1Val**2*Lambda**2*lambdaB**3*rh**7 - 59356717056*a1Val**8*a2*lambdaB**4*rh**7 -\
          127597565952*a1Val**6*a2**2*lambdaB**4*rh**7 +\
          423799382016*a1Val**4*a2**3*lambdaB**4*rh**7 -\
          31731056640*a1Val**2*a2**4*lambdaB**4*rh**7 + 955514880*a2**5*lambdaB**4*rh**7 +\
          24597872640*a1Val**8*Lambda*lambdaB**4*rh**7 -\
          310928523264*a1Val**6*a2*Lambda*lambdaB**4*rh**7 +\
          267509993472*a1Val**4*a2**2*Lambda*lambdaB**4*rh**7 -\
          22236954624*a1Val**2*a2**3*Lambda*lambdaB**4*rh**7 +\
          955514880*a2**4*Lambda*lambdaB**4*rh**7 - 7397596200960*a1Val**8*a2**3*lambdaB**5*rh**7 +\
          14447703490560*a1Val**6*a2**4*lambdaB**5*rh**7 +\
          56693882880*a1Val**4*a2**5*lambdaB**5*rh**7 + 88128*a1Val**3*lambdaB**2*rh**8 +\
          312982272*a1Val**7*lambdaB**3*rh**8 + 2495366784*a1Val**5*a2*lambdaB**3*rh**8 +\
          1832564736*a1Val**3*a2**2*lambdaB**3*rh**8 - 281677824*a1Val*a2**3*lambdaB**3*rh**8 +\
          1429799040*a1Val**5*Lambda*lambdaB**3*rh**8 +\
          2798281728*a1Val**3*a2*Lambda*lambdaB**3*rh**8 -\
          393652224*a1Val*a2**2*Lambda*lambdaB**3*rh**8 +\
          624112128*a1Val**3*Lambda**2*lambdaB**3*rh**8 -\
          109983744*a1Val*a2*Lambda**2*lambdaB**3*rh**8 +\
          26435911680*a1Val**7*a2**2*lambdaB**4*rh**8 +\
          492548677632*a1Val**5*a2**3*lambdaB**4*rh**8 -\
          98909061120*a1Val**3*a2**4*lambdaB**4*rh**8 + 4777574400*a1Val*a2**5*lambdaB**4*rh**8 -\
          123998625792*a1Val**7*a2*Lambda*lambdaB**4*rh**8 +\
          487993061376*a1Val**5*a2**2*Lambda*lambdaB**4*rh**8 -\
          68097687552*a1Val**3*a2**3*Lambda*lambdaB**4*rh**8 +\
          4299816960*a1Val*a2**4*Lambda*lambdaB**4*rh**8 +\
          7424350617600*a1Val**7*a2**4*lambdaB**5*rh**8 -\
          4317653237760*a1Val**5*a2**5*lambdaB**5*rh**8 - 6208992*a1Val**4*lambdaB**2*rh**9 -\
          1926720*a1Val**2*a2*lambdaB**2*rh**9 - 1217664*a1Val**2*Lambda*lambdaB**2*rh**9 +\
          660586752*a1Val**6*a2*lambdaB**3*rh**9 - 1544313600*a1Val**4*a2**2*lambdaB**3*rh**9 -\
          1570019328*a1Val**2*a2**3*lambdaB**3*rh**9 + 79626240*a2**4*lambdaB**3*rh**9 +\
          1082578176*a1Val**6*Lambda*lambdaB**3*rh**9 +\
          2013451776*a1Val**4*a2*Lambda*lambdaB**3*rh**9 -\
          2241091584*a1Val**2*a2**2*Lambda*lambdaB**3*rh**9 +\
          127401984*a2**3*Lambda*lambdaB**3*rh**9 +\
          1472608512*a1Val**4*Lambda**2*lambdaB**3*rh**9 -\
          671569920*a1Val**2*a2*Lambda**2*lambdaB**3*rh**9 +\
          47775744*a2**2*Lambda**2*lambdaB**3*rh**9 +\
          132624138240*a1Val**6*a2**3*lambdaB**4*rh**9 -\
          355531161600*a1Val**4*a2**4*lambdaB**4*rh**9 +\
          12368609280*a1Val**2*a2**5*lambdaB**4*rh**9 +\
          242631106560*a1Val**6*a2**2*Lambda*lambdaB**4*rh**9 -\
          276084080640*a1Val**4*a2**3*Lambda*lambdaB**4*rh**9 +\
          10139074560*a1Val**2*a2**4*Lambda*lambdaB**4*rh**9 -\
          4013162496000*a1Val**6*a2**5*lambdaB**5*rh**9 - 3280608*a1Val**5*lambdaB**2*rh**10 -\
          19022400*a1Val**3*a2*lambdaB**2*rh**10 - 1161216*a1Val*a2**2*lambdaB**2*rh**10 -\
          15829344*a1Val**3*Lambda*lambdaB**2*rh**10 - 2605824*a1Val*a2*Lambda*lambdaB**2*rh**10 -\
          1261440*a1Val*Lambda**2*lambdaB**2*rh**10 - 1973134080*a1Val**5*a2**2*lambdaB**3*rh**10 -\
          5141366784*a1Val**3*a2**3*lambdaB**3*rh**10 + 379883520*a1Val*a2**4*lambdaB**3*rh**10 -\
          687508992*a1Val**5*a2*Lambda*lambdaB**3*rh**10 -\
          7882237440*a1Val**3*a2**2*Lambda*lambdaB**3*rh**10 +\
          588570624*a1Val*a2**3*Lambda*lambdaB**3*rh**10 +\
          819535104*a1Val**5*Lambda**2*lambdaB**3*rh**10 -\
          2720853504*a1Val**3*a2*Lambda**2*lambdaB**3*rh**10 +\
          214659072*a1Val*a2**2*Lambda**2*lambdaB**3*rh**10 -\
          201222144000*a1Val**5*a2**4*lambdaB**4*rh**10 +\
          26621706240*a1Val**3*a2**5*lambdaB**4*rh**10 -\
          214944399360*a1Val**5*a2**3*Lambda*lambdaB**4*rh**10 +\
          21127495680*a1Val**3*a2**4*Lambda*lambdaB**4*rh**10 +\
          902961561600*a1Val**5*a2**6*lambdaB**5*rh**10 + 1548*a1Val**2*lambdaB*rh**11 +\
          309456*a1Val**4*a2*lambdaB**2*rh**11 - 7188480*a1Val**2*a2**2*lambdaB**2*rh**11 +\
          1492992*a2**3*lambdaB**2*rh**11 - 5209200*a1Val**4*Lambda*lambdaB**2*rh**11 -\
          16869888*a1Val**2*a2*Lambda*lambdaB**2*rh**11 + 2896128*a2**2*Lambda*lambdaB**2*rh**11 -\
          8862336*a1Val**2*Lambda**2*lambdaB**2*rh**11 + 1605888*a2*Lambda**2*lambdaB**2*rh**11 +\
          202752*Lambda**3*lambdaB**2*rh**11 - 982167552*a1Val**4*a2**3*lambdaB**3*rh**11 +\
          1000857600*a1Val**2*a2**4*lambdaB**3*rh**11 -\
          3659143680*a1Val**4*a2**2*Lambda*lambdaB**3*rh**11 +\
          1548177408*a1Val**2*a2**3*Lambda*lambdaB**3*rh**11 -\
          2115763200*a1Val**4*a2*Lambda**2*lambdaB**3*rh**11 +\
          570654720*a1Val**2*a2**2*Lambda**2*lambdaB**3*rh**11 +\
          82957271040*a1Val**4*a2**5*lambdaB**4*rh**11 +\
          71212400640*a1Val**4*a2**4*Lambda*lambdaB**4*rh**11 + 17724*a1Val**3*lambdaB*rh**12 +\
          6696*a1Val*a2*lambdaB*rh**12 + 5856*a1Val*Lambda*lambdaB*rh**12 +\
          19184256*a1Val**3*a2**2*lambdaB**2*rh**12 + 6829056*a1Val*a2**3*lambdaB**2*rh**12 +\
          18648000*a1Val**3*a2*Lambda*lambdaB**2*rh**12 +\
          13398912*a1Val*a2**2*Lambda*lambdaB**2*rh**12 +\
          1545120*a1Val**3*Lambda**2*lambdaB**2*rh**12 +\
          7568640*a1Val*a2*Lambda**2*lambdaB**2*rh**12 + 976896*a1Val*Lambda**3*lambdaB**2*rh**12 +\
          2159308800*a1Val**3*a2**4*lambdaB**3*rh**12 +\
          3474579456*a1Val**3*a2**3*Lambda*lambdaB**3*rh**12 +\
          1357378560*a1Val**3*a2**2*Lambda**2*lambdaB**3*rh**12 +\
          39624*a1Val**2*a2*lambdaB*rh**13 + 5328*a2**2*lambdaB*rh**13 +\
          37584*a1Val**2*Lambda*lambdaB*rh**13 + 9696*a2*Lambda*lambdaB*rh**13 +\
          4368*Lambda**2*lambdaB*rh**13 + 16137216*a1Val**2*a2**3*lambdaB**2*rh**13 +\
          33443712*a1Val**2*a2**2*Lambda*lambdaB**2*rh**13 +\
          20904768*a1Val**2*a2*Lambda**2*lambdaB**2*rh**13 +\
          3548160*a1Val**2*Lambda**3*lambdaB**2*rh**13 + a1Val*rh**14 +\
          22032*a1Val*a2**2*lambdaB*rh**14 + 41904*a1Val*a2*Lambda*lambdaB*rh**14 +\
          19896*a1Val*Lambda**2*lambdaB*rh**14 + a2*rh**15 + Lambda*rh**15)/\
          (a1Val**5*lambdaB**5*rh**5*(2 + a1Val*rh)**5)
    return a7Val
################### Fit Value for a2 ######################################################
def error_a2(a2, rh, lambdaB, LambdaB, r_vals, NSConti_interp):
    f_vals = np.array([fNH(r, lambdaB, LambdaB, rh, a2)[0] for r in r_vals])  # Evalúa f(r)
    num_sol_vals = NSConti_interp(r_vals)  
    return np.mean((f_vals - num_sol_vals) ** 2)  

#Values near rh
def generate_r_vals(rh, delta=1e-3, num_points=100):
   
    return np.linspace(rh, rh + delta, num_points)

# Optimization for a2
def optimize_a2(rh, lambdaB, LambdaB, NSConti_interp, delta=0.1, num_points=100, a2_initial=0.1):
    r_vals = generate_r_vals(rh, delta, num_points)
    
    result = minimize(error_a2, x0=a2_initial, args=(rh, lambdaB, LambdaB, r_vals, NSConti_interp))
    
    return result.x[0]

############ Whats the mass with this rh ###############################
def MassValue(rh, lamB, LambdaB):
    a1Val = a1(rh,lamB,LambdaB)
    MassAdi= 1 + (12*a1Val**2*lamB)/(Mp**5*rh**2) + (4*a1Val**3*lamB)/(Mp**5*rh) - (LambdaB*rh**2)/3.
    Mass = rh*Mp**2*MassAdi/(2*MSol)
    return print('The Mass is',Mass, 'Solar Masses')


##########################################
#### Some Important Functions ############
##########################################

####Kottler with Modified Mass for cubic gravity ########
def Kottler(r,Lam,lam):
    return 1-(1/r)*(3/(16*lam*Lam**2+3)) - r**2*Lam/3
#### Sch with modified mass for cubic gravity ##########
def Sch(r,lam,Lam):
    return 1-(1/r)*(3/(16*lam*Lam**2+3))
## Ricci Scalar for a f(r) metric and taking f'' as the system of DOE ###
RicciScalar =  lambda r, y, y1, Lambda, lamb:(r**3*(3 + r*(-3 + Lambda*r**2)) + 3*(y*(48*lamb + r**4 + 48*lamb*(-2 + y)*y) + 96*lamb*r*(-1 + y)*y*y1\
            - 12*lamb*r**2*(1 + 4*y)*y1**2 - 4*lamb*r**3*y1**3))/(36.*lamb*r**2*y*(2 - 2*y + r*y1))

############################################
############### Plots ######################
############################################

def evaluate_functions(func1, func2, r_values, lamb, Lambda, tol=0.75):
    """
    func1 -> Compared Function
    func2 -> Reference Function (Numerical solution interpolated)
    r_vals -> values in the range
    parms -> parameters function 1
    tol -> Tolerance value
    """
    valid_values = []
    for r in r_values:
        result1 = func1(r, lamb, Lambda)
        result2 = func2(r)
        diff = abs(result1 - result2)

        if diff > tol:
            break  # Detiene el bucle si la tolerancia se supera
    
    valid_values.append(r)  # Solo agrega valores si no se ha detenido el bucle

    return valid_values
