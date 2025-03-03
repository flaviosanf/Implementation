import numpy as np
import matplotlib.pyplot as plt

# modulos que están en la carpeta
import text_curv as tc  # para añadir texto curveado
import plt_conf as conf  # modulo de configuración de gráfico

from AngleAnnotation import AngleAnnotation
from matplotlib.patches import Circle

# cargando el módulo particular de las configuraciones

conf.general()  # cargando configuración general
#conf.latex()  # cargando conf para guardar en formato latex
def crear_imagen(ax=None):
 
    if ax is None:
        # Crear una figura y un eje si no se proporciona
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5),
                               sharex=False, sharey=False,
                               gridspec_kw=dict(hspace=0, wspace=.4))

    # Coordenadas de los puntos
    vert2 = [[0, 0.7, 1.35], [0.6, 1.4, 0.6]]  # [[x], [y]]



    # Dibujar los puntos
    plt1, = ax.plot(vert2[0], vert2[1], ls='', marker='o', markersize=6, color='#757373')  #Plot de los puntos en las intersecciones
    
    #triangle in curved space
    ax.annotate(r'', xy=(vert2[0][0], vert2[1][0]), xytext=(vert2[0][2], vert2[1][2]), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=0.1"),
                horizontalalignment='center', verticalalignment='center'
                )  #Linea inferior 
    ax.annotate(r'', xy=(vert2[0][2], vert2[1][2]), xytext=(vert2[0][1], vert2[1][1]), 
                            arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=-0.1"),
             horizontalalignment='center', verticalalignment='center'
            ) #Linea derecha
    ax.annotate(r'', xy=(vert2[0][1], vert2[1][1]), xytext=(vert2[0][0], vert2[1][0]), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=-0.1"),
             horizontalalignment='center', verticalalignment='center'
            ) #Linea Izquierda

### extensiones
    ax.annotate(r'', xy=(vert2[0][0]+0.01, vert2[1][0]),  xytext=(vert2[0][0]-0.15, vert2[1][0]-0.03), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=-0.04"),
                horizontalalignment='center', verticalalignment='center'
                )

    ax.annotate(r'', xy=(vert2[0][2]-0.01, vert2[1][2]), xytext=(vert2[0][2]+0.15, vert2[1][2]-0.03), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=-0.04"),
                horizontalalignment='center', verticalalignment='center'
                )

##
    ax.annotate(r'', xy=(vert2[0][2], vert2[1][2]), xytext=(vert2[0][2]+0.08, vert2[1][2]-0.15), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=0.05"),
             horizontalalignment='center', verticalalignment='center'
            )

    ax.annotate(r'', xy=(vert2[0][1], vert2[1][1]), xytext=(vert2[0][1]-0.1, vert2[1][1]+0.08), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=0"),
             horizontalalignment='center', verticalalignment='center'
            )

##
    ax.annotate(r'', xy=(vert2[0][1], vert2[1][1]), xytext=(vert2[0][1]+0.12, vert2[1][1]+0.08), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=0.05"),
             horizontalalignment='center', verticalalignment='center'
            )
    ax.annotate(r'', xy=(vert2[0][0], vert2[1][0]), xytext=(vert2[0][0]-0.08, vert2[1][0]-0.13), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=-0.07"),
             horizontalalignment='center', verticalalignment='center'
               )


# limit
    ax.set_ylim(-0.1, 1.6)
    ax.set_xlim(-0.1, 1.6)

##Betas

    center = [(vert2[0][0], vert2[1][0]),
             (vert2[0][2], vert2[1][2]),
              (vert2[0][1], vert2[1][1])]

    p1 = [(vert2[0][0], vert2[1][0]),
          (vert2[0][0], vert2[1][0]+3.5), 
        (vert2[0][2]-6, vert2[1][1]-1)]

    p2 = [(vert2[0][0]+2, vert2[1][0]+4.5),
          (vert2[0][0], vert2[1][0]),
          (vert2[0][2]+6, vert2[1][1]-2)]

    AngleAnnotation(center[0], p1[0], p2[0],ax=ax, size=50, text=r"$\beta_1$",
                textposition="outside",
                text_kw=dict(fontsize=12, color="#0a0101")) 
    AngleAnnotation(center[1], p1[1], p2[1],ax=ax, size=50, text=r"$\beta_2$",\
                    textposition="outside",\
                text_kw=dict(fontsize=12, color="#0a0101"))
    AngleAnnotation(center[2], p1[2], p2[2],ax=ax, size=30, text=r"$\beta_3$",\
                    textposition="outside",\
                text_kw=dict(fontsize=12, color="#0a0101"))



# Source
    x1, y1 = 0.7, 0.2
    ax.plot([x1], [y1], 'bo', alpha=0.2)  #Circulo Blanco
    el = Circle((x1, y1), radius=0.1, angle=0, alpha=0.2)
    ax.add_artist(el) #Halo de Circulo Blanco

# factor de impacto
    fx1, fy1 = 0.85, 0.72
    fx2, fy2 = 0.4, 1.15
    fx3, fy3 = 1.4, 0.3

### Linea de b1, conexion de angulo y lo escrito en ella
    ax.annotate(r'', xy=(x1-0.1, fy1-0.04), xytext=(x1+0.1, fy1-0.04), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               ) #Linea Horizontal de Factor de impacto
    ax.annotate(r'', xy=(x1, y1), xytext=(x1, fy1-0.05), 
                arrowprops=dict(arrowstyle="<|-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(x1+0.03, 0.57, r'$b_1$', fontsize=12)
    ax.text(x1-0.14, fy1-0.015, r'$\phi_{b_1}=\pi/2$', fontsize=12)

### Linea de b2
    ax.annotate(r'', xy=(0.985, 1.18), xytext=(1.125, .995), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.annotate(r'', xy=(x1, y1), xytext=(1.05, 1.08), 
                arrowprops=dict(arrowstyle="<|-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(.85, 0.5, r'$b_2$', fontsize=12)
    ax.text(0.987, 0.98, r'$\phi_{b_2}=\pi/2-\delta$', fontsize=12, rotation=-60)

##Linea de b3
    ax.annotate(r'', xy=(0.256, 1.015), xytext=(0.41, 1.185), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.annotate(r'', xy=(x1, y1), xytext=(0.35, 1.10), 
                arrowprops=dict(arrowstyle="<|-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(.515, 0.5, r'$b_3$', fontsize=12)
    ax.text(.17, 1, r'$\phi_{b_3}=\pi/2+\delta$', fontsize=13, rotation=52)

# text
    ax.text(0, 0.05, r'$\mathcal{M}_{\mathrm{opt}}$ space', fontsize=12)
    ax.text(0, -0.05, r'Equatorial Plane', fontsize=12)

### Eje x
    ax.annotate(r'$x$', xy=(x1, y1), xytext=(x1+0.9, y1), 
                arrowprops=dict(arrowstyle="<-", facecolor='black', lw=1., ls='-.',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )


    ax.text(vert2[0][0]-0.4, vert2[1][0]-0.1, r'$\pi-\phi_{\mathrm{Min}}=\phi_1$', fontsize=12, rotation=1,
            transform_rotates_text=True)
    ax.text(vert2[0][2]+0.1, vert2[1][2]-0.1, r'$\phi_2=\phi_{\mathrm{Min}}$', fontsize=12, rotation=-1,
           transform_rotates_text=True)
    ax.text(vert2[0][1]-0.1, vert2[1][1]+0.12, r'$\phi_3=\pi/2$', fontsize=12, rotation=0)

# angle
    x3, y3 = 0.7, 0.2
    ax.annotate(r'', xy=(x3+0.14, y3-0.01), xytext=(x3-0.15, y3+0.05), 
                arrowprops=dict(arrowstyle="<-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=-0.8"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(x3-0.2, y3, r'$\phi$', fontsize=15)

    ax.text(-0.1, 0.40, r'$\mathcal{C}_1$', fontsize=12)
    ax.text(.83, 1.47, r'$\mathcal{C}_3$', fontsize=12)
    ax.text(1.42, 0.4, r'$\mathcal{C}_2$', fontsize=12)

    ax.axis('off')


    plt2, = ax.plot([], [], ls='-', c='k')
    plt3, = ax.plot([], [], ls=':', c='k')
    ax.legend(loc=(0.73, 0.8), handles=[plt1, plt3, plt2], labels=[r'Vertex',
                                                  r' $b_{p}$ impact parameters',
                                                  r'$\mathcal{C}_{p}$ geodesics'],
              frameon=False, fontsize='small', labelspacing=0.8, handlelength=1.2)

    return ax




def crear_imagen2(ax=None):
 
    if ax is None:
        # Crear una figura y un eje si no se proporciona
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5),
                               sharex=False, sharey=False,
                               gridspec_kw=dict(hspace=0, wspace=.4))

    # Coordenadas de los puntos
    vert2 = [[0, 0.7, 1.35], [0.6, 1.4, 0.6]]  # [[x], [y]]



    # Dibujar los puntos
    plt1, = ax.plot(vert2[0], vert2[1], ls='', marker='o', markersize=6, color='#757373')  #Plot de los puntos en las intersecciones
    
    #triangle in curved space
    ax.annotate(r'', xy=(vert2[0][0], vert2[1][0]), xytext=(vert2[0][2], vert2[1][2]), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=0.1"),
                horizontalalignment='center', verticalalignment='center'
                )  #Linea inferior 
    ax.annotate(r'', xy=(vert2[0][2], vert2[1][2]), xytext=(vert2[0][1], vert2[1][1]), 
                            arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=-0.1"),
             horizontalalignment='center', verticalalignment='center'
            ) #Linea derecha
    ax.annotate(r'', xy=(vert2[0][1], vert2[1][1]), xytext=(vert2[0][0], vert2[1][0]), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=-0.1"),
             horizontalalignment='center', verticalalignment='center'
            ) #Linea Izquierda

### extensiones
    ax.annotate(r'', xy=(vert2[0][0]+0.01, vert2[1][0]),  xytext=(vert2[0][0]-0.15, vert2[1][0]-0.03), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=-0.04"),
                horizontalalignment='center', verticalalignment='center'
                )

    ax.annotate(r'', xy=(vert2[0][2]-0.01, vert2[1][2]), xytext=(vert2[0][2]+0.15, vert2[1][2]-0.03), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=-0.04"),
                horizontalalignment='center', verticalalignment='center'
                )

##
    ax.annotate(r'', xy=(vert2[0][2], vert2[1][2]), xytext=(vert2[0][2]+0.08, vert2[1][2]-0.15), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=0.05"),
             horizontalalignment='center', verticalalignment='center'
            )

    ax.annotate(r'', xy=(vert2[0][1], vert2[1][1]), xytext=(vert2[0][1]-0.1, vert2[1][1]+0.08), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3,rad=0"),
             horizontalalignment='center', verticalalignment='center'
            )

##
    ax.annotate(r'', xy=(vert2[0][1], vert2[1][1]), xytext=(vert2[0][1]+0.12, vert2[1][1]+0.08), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=0.05"),
             horizontalalignment='center', verticalalignment='center'
            )
    ax.annotate(r'', xy=(vert2[0][0], vert2[1][0]), xytext=(vert2[0][0]-0.08, vert2[1][0]-0.13), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=-0.07"),
             horizontalalignment='center', verticalalignment='center'
               )


# limit
    ax.set_ylim(-0.1, 1.6)
    ax.set_xlim(-0.1, 1.6)

##Betas

    center = [(vert2[0][0], vert2[1][0]),
             (vert2[0][2], vert2[1][2]),
              (vert2[0][1], vert2[1][1])]

    p1 = [(vert2[0][0], vert2[1][0]),
          (vert2[0][0], vert2[1][0]+3.5), 
        (vert2[0][2]-6, vert2[1][1]-1)]

    p2 = [(vert2[0][0]+2, vert2[1][0]+4.5),
          (vert2[0][0], vert2[1][0]),
          (vert2[0][2]+6, vert2[1][1]-2)]

    AngleAnnotation(center[0], p1[0], p2[0],ax=ax, size=50, text=r"$\beta_1$",
                textposition="outside",
                text_kw=dict(fontsize=12, color="#0a0101")) 
    AngleAnnotation(center[1], p1[1], p2[1],ax=ax, size=50, text=r"$\beta_2$",\
                    textposition="outside",\
                text_kw=dict(fontsize=12, color="#0a0101"))
    AngleAnnotation(center[2], p1[2], p2[2],ax=ax, size=30, text=r"$\beta_3$",\
                    textposition="outside",\
                text_kw=dict(fontsize=12, color="#0a0101"))

# Source
    x1, y1 = 0.7, 0.2
    ax.plot([x1], [y1], 'bo', alpha=0.2)  #Circulo Blanco
    el = Circle((x1, y1), radius=0.1, angle=0, alpha=0.2)
    ax.add_artist(el) #Halo de Circulo Blanco

# factor de impacto
    fx1, fy1 = 0.85, 0.72
    fx2, fy2 = 0.4, 1.15
    fx3, fy3 = 1.4, 0.3

### Linea de b1, conexion de angulo y lo escrito en ella
    ax.annotate(r'', xy=(x1-0.1, fy1-0.04), xytext=(x1+0.1, fy1-0.04), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               ) #Linea Horizontal de Factor de impacto
    ax.annotate(r'', xy=(x1, y1), xytext=(x1, fy1-0.05), 
                arrowprops=dict(arrowstyle="<|-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(x1+0.03, 0.57, r'$b_1$', fontsize=13)
    ax.text(x1-0.14, fy1+0.01, r'$\phi_{b_1}=\pi/2$', fontsize=15)

### Linea de b2
    ax.annotate(r'', xy=(0.985, 1.18), xytext=(1.125, .995), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.annotate(r'', xy=(x1, y1), xytext=(1.05, 1.08), 
                arrowprops=dict(arrowstyle="<|-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(.87, 0.5, r'$b_2$', fontsize=13)
    ax.text(0.95, 0.96, r'$\phi_{b_2}=\pi/2-\delta$', fontsize=14.5, rotation=-52)

##Linea de b3
    ax.annotate(r'', xy=(0.256, 1.015), xytext=(0.41, 1.185), 
                arrowprops=dict(arrowstyle="-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.annotate(r'', xy=(x1, y1), xytext=(0.35, 1.10), 
                arrowprops=dict(arrowstyle="<|-", facecolor='black', lw=1.5, ls=':',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(.5, 0.5, r'$b_3$', fontsize=13)
    ax.text(.1, .95, r'$\phi_{b_3}=\pi/2+\delta$', fontsize=15, rotation=45)

# text
    ax.text(0, 0.05, r'$\mathcal{M}_{\mathrm{opt}}$ space', fontsize=12)
    ax.text(0, -0.05, r'Equatorial Plane', fontsize=12)

### Eje x
    ax.annotate(r'$x$', xy=(x1, y1), xytext=(x1+0.9, y1), 
                arrowprops=dict(arrowstyle="<-", facecolor='black', lw=1., ls='-.',
                               connectionstyle="arc3, rad=0"),
             horizontalalignment='center', verticalalignment='center'
               )


    ax.text(vert2[0][0]-0.43, vert2[1][0]-0.1, r'$\pi-\phi_{\mathrm{Min}}=\phi_1$', fontsize=12, rotation=1,
            transform_rotates_text=True)
    ax.text(vert2[0][2]+0.1, vert2[1][2]-0.1, r'$\phi_2=\phi_{\mathrm{Min}}$', fontsize=12, rotation=-1,
           transform_rotates_text=True)
    ax.text(vert2[0][1]-0.1, vert2[1][1]+0.12, r'$\phi_3=\pi/2$', fontsize=12, rotation=0)

# angle
    x3, y3 = 0.7, 0.2
    ax.annotate(r'', xy=(x3+0.14, y3-0.01), xytext=(x3-0.15, y3+0.05), 
                arrowprops=dict(arrowstyle="<-", facecolor='black', lw=1.5, ls='-',
                               connectionstyle="arc3, rad=-0.8"),
             horizontalalignment='center', verticalalignment='center'
               )
    ax.text(x3-0.2, y3, r'$\phi$', fontsize=16)

    ax.text(-0.1, 0.40, r'$\mathcal{C}_1$', fontsize=12)
    ax.text(.83, 1.47, r'$\mathcal{C}_3$', fontsize=12)
    ax.text(1.42, 0.4, r'$\mathcal{C}_2$', fontsize=12)

    ax.axis('off')


    plt2, = ax.plot([], [], ls='-', c='k')
    plt3, = ax.plot([], [], ls=':', c='k')
    ax.legend(loc=(0.73, 0.8), handles=[plt1, plt3, plt2], labels=[r'Vertex',
                                                  r' $b_{p}$ impact parameters',
                                                  r'$\mathcal{C}_{p}$ geodesics'],
              frameon=False, fontsize='small', labelspacing=0.8, handlelength=1.2)

    return ax

