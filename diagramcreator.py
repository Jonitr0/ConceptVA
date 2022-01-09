

import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, LabelSet


def create_diagram(diagram_average_below_salicity,
                   diagram_average_above_salicity,
                   diagram_min_below_salicity,
                   diagram_min_above_salicity,
                   diagram_max_below_salicity,
                   diagram_max_above_salicity,
                   col_names,
                   col_to_global_max,
                   col_to_global_min):

    num_vars = len(col_names)
    centre = 0.5

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def unit_poly_verts(theta, centre):
        """Return vertices of polygon for subplot axes.
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [centre] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def radar_patch(r, theta, centre):
        yt = (r*centre + 0.01) * np.sin(theta) + 0.5
        xt = (r*centre + 0.01) * np.cos(theta) + 0.5
        return xt, yt

    verts = unit_poly_verts(theta, centre)
    x = [v[0] for v in verts]
    y = [v[1] for v in verts]

    p = figure(title="Data ")
    text = col_names
    source = ColumnDataSource({'x':x + [centre],'y':y + [1],'text':text})

    p.line(x="x", y="y", source=source)

    labels = LabelSet(x="x",y="y",text="text",source=source)

    p.add_layout(labels)

    def diagram_dict_to_np_array(diagram_dict):
        result_list = list()
        for col in col_names:
            scaled_max = col_to_global_max[col] - col_to_global_min[col]
            scaled_output = (diagram_dict[col] - col_to_global_min[col]) / scaled_max * 0.5
            result_list.append(scaled_output)
        return np.array(result_list)

    np_average_below_salicity = diagram_dict_to_np_array(diagram_average_below_salicity)
    np_average_above_salicity =  diagram_dict_to_np_array(diagram_average_below_salicity)
    np_min_below_salicity = diagram_dict_to_np_array(diagram_min_below_salicity)
    np_min_above_salicity = diagram_dict_to_np_array(diagram_min_above_salicity)
    np_max_below_salicity = diagram_dict_to_np_array(diagram_max_below_salicity)
    np_max_above_salicity = diagram_dict_to_np_array(diagram_max_above_salicity)

    # example factor:
    f1 = np.array([0.88, 0.01, 0.03, 0.03, 1., 0.06, 0.01]) * 0.5
    f2 = np.array([0.07, 0.95, 0.04, 0.05, 1., 0.02, 0.01]) * 0.5
    f3 = np.array([0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00]) * 0.5
    f4 = np.array([0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98]) * 0.5
    #f5 = np.array([0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]) * 0.5
    #xt = np.array(x)

    flist = [f1, f2, f3, f4]

    '''    
    flist = [np_average_below_salicity,
             np_average_above_salicity,
             np_min_below_salicity,
             np_min_above_salicity,
             np_max_below_salicity,
             np_max_above_salicity]
    '''
    colors = ['blue','green','red', 'orange','purple']
    for i in range(len(flist)):
        xt, yt = radar_patch(flist[i], theta, centre)
        p.patch(x=xt, y=yt[0], fill_alpha=0.15, fill_color=colors[i])
    save(p)

def create_diagram2(diagram_average_below_salicity,
                   diagram_average_above_salicity,
                   diagram_min_below_salicity,
                   diagram_min_above_salicity,
                   diagram_max_below_salicity,
                   diagram_max_above_salicity,
                   col_names,
                   col_to_global_max,
                   col_to_global_min):

    num_vars = len(col_names)

    centre = 0.5

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def unit_poly_verts(theta, centre):
        """Return vertices of polygon for subplot axes.
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [centre] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def radar_patch(r, theta, centre ):
        """ Returns the x and y coordinates corresponding to the magnitudes of
        each variable displayed in the radar plot
        """
        # offset from centre of circle
        offset = 0.01
        yt = (r*centre + offset) * np.sin(theta) + centre
        xt = (r*centre + offset) * np.cos(theta) + centre
        return xt, yt


    def diagram_dict_to_np_array(diagram_dict):
        result_list = list()
        for col in col_names:
            scaled_max = col_to_global_max[col] - col_to_global_min[col]
            scaled_output = (diagram_dict[col] - col_to_global_min[col]) / scaled_max
            result_list.append(scaled_output)
        return np.array(result_list)


    verts = unit_poly_verts(theta, centre)
    x = [v[0] for v in verts] + [verts[0][0]]
    y = [v[1] for v in verts] + [verts[0][1]]

    p = figure(title="Baseline - Radar plot",plot_width=500, plot_height=500)
    source = ColumnDataSource({'x':x + [centre ],'y':y + [1],'text':col_names})

    #p.line(x="x", y="y", source=source)
    p.patch(x='x', y='y', fill_alpha=0.0, source=source, line_width=1.5,color="black")

    labels = LabelSet(x="x",y="y",text="text",source=source)

    p.add_layout(labels)

    np_average_below_salicity = diagram_dict_to_np_array(diagram_average_below_salicity)
    np_average_above_salicity =  diagram_dict_to_np_array(diagram_average_above_salicity)
    np_min_below_salicity = diagram_dict_to_np_array(diagram_min_below_salicity)
    np_min_above_salicity = diagram_dict_to_np_array(diagram_min_above_salicity)
    np_max_below_salicity = diagram_dict_to_np_array(diagram_max_below_salicity)
    np_max_above_salicity = diagram_dict_to_np_array(diagram_max_above_salicity)
    # example factor:
    f1 = np.array([0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01])
    f2 = np.array([0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01])
    f3 = np.array([0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00])
    f4 = np.array([0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98])
    f5 = np.array([0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00])
    #xt = np.array(x)
    flist = [np_average_below_salicity,
             np_average_above_salicity]
             #np_min_below_salicity,
             #np_min_above_salicity,
             #np_max_below_salicity,
             #np_max_above_salicity]
    colors = ['black','red','blue', 'green', 'yellow', 'purple']
    labels = ['avg_not_sal','avg_sal']#,'min_not_sal','min_sal','max_not_sal','max_sal']
    print(flist)
    for i in range(len(flist)):
        xt, yt = radar_patch(flist[i], theta, centre)
        p.patch(x=xt, y=yt, fill_alpha=0.0, line_color=colors[i],legend=labels[i])
        #p.line(x=xt, y=yt, color=colors[i])
    save(p)
