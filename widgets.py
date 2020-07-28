#%%
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
import utils
import numpy as np
import plotly.offline as py
py.init_notebook_mode()
from IPython import display
#%%
up_shift = utils.FCR_bac([0.1, 1.2])
up_shift.integ_sigma(t=np.linspace(0, 5, 500))
trace = go.Scatter(x=up_shift.t, y=up_shift.sigma_t,
                   line=dict(dash='dash', color='yellowgreen', width=3))
g = go.Figure()
g.add_trace(trace)
g = go.FigureWidget(g)

def upate(lambdai, lambdaf):
    lambdas = [lambdai, lambdaf]
    new_line = utils.FCR_bac[lambdas]
    new_line.integ_sigma(t=np.linspace(0, 5, 500))
    g.data[0].x, g.data[0].y = new_line.t, new_line.sigma_t

slid1 = widgets.FloatSlider(value=0.1, min=0.001, max=3., step=0.01)
slid2 = widgets.FloatSlider(value=1.2, min=0.001, max=3., step=0.01)
container = widgets.HBox([slid1, slid2])
widgets.interactive(upate, lambdai=slid1, lambdaf=slid2)
final = widgets.VBox([container, g])
display(final)

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector

def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
    print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print('used button  : ', eclick.button)

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.ES.active:
        print('EllipseSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.ES.active:
        print('EllipseSelector activated.')
        toggle_selector.ES.set_active(True)

x = np.arange(100.) / 99
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)

toggle_selector.ES = EllipseSelector(ax, onselect, drawtype='line')
fig.canvas.mpl_connect('key_press_event', toggle_selector)
plt.show()