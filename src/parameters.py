# --------- PATHS ---------

DATA_PATH = '../data/'
DATA_PATH_RAW = '../data/raw/'
PLOT_PATH = '../plots/'
PACKAGE_PATH = '/home/monicaconte/nica/phd/Projects/Efficient_ML_Memristor_Network/codes/ahkab'
# PACKAGE_PATH = '/Users/monicaconte/PhD/Projects/Contrastive_Learning_Memristors/Learning_Neural_Networks_true/codes/ahkab'

# --------- NETWORKS PLOTS STYLES ---------

color_dots = ['forestgreen','darkslateblue','silver'] # color of the three types of nodes: input, output, hidden
font_size_nw = 12 #size font on nodes
nodes_size = 1000 #size nodes
width = 5 #width connecting edges

color_edges = 'lightblue'
color_font_edges = 'white'

# --------- PLOTS STYLES ---------

figsize_2horizontal = (11,5)
figsize_3horizontal = (13,5)
figsize_4horizontal = (13,5)
figsize_1 = (4.4,4)

legend_size = 15
axis_fontsize = 19
size_ticks = 20
marker_size = 10

# -> MSE style

from pypalettes import load_cmap
cmap = load_cmap("Callanthias_australis")

weight_styles = {
    'length': dict(marker = 'o', c = cmap.colors[0], lw=1, label = rf'$L$', ylabel_weights = r'$L$[$\mu$ m]'),
    'rho': dict(marker = 'o', c = cmap.colors[1], lw=1, label = rf'$\rho$', ylabel_weights = r'$\rho$[mM]'),
    'radius_base': dict(marker = 'o', c = cmap.colors[2], lw=1, label = rf'$R_b$', ylabel_weights = r'$R_b$[nm]'),
    'pressure': dict(marker = 'o', c = cmap.colors[3], lw=1, label = rf'$P$', ylabel_weights = r'$P$[bar]'),
    'resistance': dict(marker = 'o', c = 'mediumblue', lw=1, label = rf'$R$', ylabel_weights = r'$R$[$\Omega$]'),
    'length_radius_base': dict(marker = 'o', c = 'mediumblue', lw=1, label = rf'$L/R_b$', ylabel_weights = r'$R$[$\Omega$]'),
    'length_pressure': dict(marker = 'o', c = 'dodgerblue', lw=1, label = rf'$L/P$', ylabel_weights = r'$R$[$\Omega$]'),
    'length_var': dict(marker = 'o', c = 'blueviolet', lw=1, label = r'$L_{var}$', ylabel_weights = r'$L$[$\mu$ m]'),
    'pressure_var': dict(marker = 'o', c = 'darkgreen', lw=1, label = r'$P_{var}$', ylabel_weights = r'$L$[$\mu$ m]'),

}

memr_resistances_style = dict(marker = 'o', markersize=3, lw=1)

# REGRESSION TRAINING

regression_styles = {
    'length': dict(marker = 'o', s=30, c = cmap.colors[0], lw=0.6),
    'length_des': dict(c = 'lightpink', lw=2, label = rf'$V^D$'),
    'rho': dict(marker = 'o', s=30, c = cmap.colors[1], lw=1,),
    'rho_des': dict(c = 'moccasin', lw=2, label = rf'$V^D$'),
    'radius_base': dict(marker = 'o', c = cmap.colors[2], lw=1, label = rf'$R_b$'),
    'radius_base_des': dict(c = 'yellow', lw=4, label = rf'$V^D$'),
    'pressure': dict(marker = 'o', c = cmap.colors[3], lw=1, label = rf'$P$'),
    'pressure_des': dict(c = 'green', lw=4, label = rf'$V^D$')
}

reg_desired = dict(c = 'lightblue', lw=3, label = rf'$V_1^D$')
reg_output = dict(c = 'mediumblue', marker='o', label = rf'$V_1$')

# CHECKING DYNAMICS VD

potential_drops_style = {
    'v1': dict(c = color_dots[0], lw=3,  label = rf'$V_1$'),
    'v3': dict(c = color_dots[0], lw=3, label = rf'$V_3$'),
    'deltav1': dict(c = 'plum', lw=5, label = rf'$\Delta V_1$'),
    'deltav2': dict(c = 'darkorchid', lw=2, label = rf'$\Delta V_2$')}

conductances_style = {
    'M1': dict(c = 'darkcyan', lw=5, label=r'$M_1$'),
    'M2': dict(c = 'lightblue', lw=2, label=r'$M_2$')
    
}

lengths_style = {
    'M1': dict(c = 'green', lw=5, label=r'$M_1$'),
    'M2': dict(c = 'mediumseagreen', lw=2, label=r'$M_2$')
    
}

steady_state_point = {    
    'M1': dict(marker = '^', c = 'darkcyan', s=200, edgecolors= 'blue' , label=r'steady state $M_1$'),
    'M2': dict(marker = 'o', c = 'lightblue', s=200, edgecolors= 'blue', label=r'steady state $M_2$')
    }

g_infinity_style = {
    'M1': dict(c = 'mediumaquamarine', lw=5, label=r'$g_{\infty}^{M_1}$'),
    'M2': dict(c = 'dodgerblue', lw=2, label=r'$g_{\infty}^{M_2}$')
}


