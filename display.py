# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 2020
@author: Violeta

Skeleton data processing for spatial reconstruction

Visualization functions for:
    - Individual tracks 2D: raw data and from origin of coordinates
    - Individual tracks 3D: raw data and from origin of coordinates
    - Set of tracks 2D from different perspectives to compare the motion
    - Others

"""


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as ani
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
import pylab as pl
from matplotlib import collections  as mc
from matplotlib.patches import Wedge, RegularPolygon
import pandas as pd

import dataprocessing as dp

def display_2d(trajectory_data):
    """
    Show coordinates in 2D space, raw data

    :param trajectory_data: dataframe with trajectory and bodies detected data
    :return: plot
    """
    sns.set()
    sns.lmplot(x="x", y="y", data=trajectory_data, fit_reg=False, hue='ID_subject', legend=False)
    plt.ylabel('(Y) Distance from Kinect')
    plt.xlabel('(X) Left/Right distance from Kinect Center')
    plt.title('Kinect Trajectory: ')
    plt.legend(loc='upper center')

    return plt


def display_2d_global(trajectory_data):
    """
    Show coordinates in 2D space, raw data

    :param trajectory_data: dataframe with trajectory and bodies detected data
    :return:
    """
    sns.set()
    sns.lmplot(x="x", y="y", data=trajectory_data, fit_reg=False, hue='ID_subject', legend=False)
    plt.ylabel('(Y) Distance from Kinect')
    plt.xlabel('(X) Left/Right distance from Kinect Center')
    plt.title('Kinect Trajectories: ')
    plt.legend(loc='upper center')

    return plt


def display_2d_origin(trajectory_data, type='ID_exp'):
    """
    Show coordinates in 2D space from origin
    :param type:
    :param trajectory_data: dataframe with trajectory and bodies detected data
    :return: plot
    """
    sns.set()
    sns.lmplot(x="origin_x", y="origin_y", data=trajectory_data, fit_reg=False, hue=type, legend=False)
    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')
    plt.title('Kinect Trajectory: '+ trajectory_data['ID_exp'][0])
    plt.legend(loc='lower left')

    return plt


def display_2d_origin_global(trajectory_data, control_points,type='ID_exp'):

    """
    Show coordinates in 2D space from origin with control points
    :param trajectory_data: dataframe with trajectory and bodies detected data
    :param control_points: points with real world coordinates in which people stopped
    :return: plot
    """
    sns.set()

    #sns.set_palette("CMRmap")
    #sns.set_palette("rainbow")
    sns.scatterplot(x="origin_x", y="origin_y", data=trajectory_data,  hue='ID_exp', style='ID_exp', legend='brief', alpha=0.9,palette="deep")
    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')
    plt.title('Kinect Trajectories by '+ type +' ' )
    sns.scatterplot(x='origin_x', y='origin_y', data=control_points, hue='description', style='description', palette='twilight', s=100, legend=False)

    plt.legend(loc='lower left')

    return plt


def display_control_points(data_setup):
    """
    Show points of control for different measurements:
    Ex: Static points such as floor markers
    :param data_setup: x and y coordinates data frame
    :return: plot object
    """
    sns.set()
    sns.pointplot(data=data_setup, x="x", y="y", join=False, color='red')
    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')
    plt.title('Control Points')

    return plt


def display_body_direction_stops_all(data_kinect, body_data, control_points):
    """Display body orientation with stops

    :param data_kinect: dataframe with trajectory and bodies detected data
    :param body_data:
    :param control_points: points with real world coordinates in which people stopped
    :return: plot
    """

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    U = np.cos((data_kinect['re_body_angle']) )
    V = np.sin((data_kinect['re_body_angle']) )

    sns.set()

    norm = matplotlib.colors.Normalize()
    norm.autoscale(data_kinect['re_body_angle'])
    cm = matplotlib.cm.viridis

    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots()

   
    #sns.scatterplot(x='x', y='y', data=control_points, hue='description', style='description', palette='twilight', s=500, legend=False)
    ax.quiver(data_kinect['shl_x'], data_kinect['shl_y'], U, V, angles=data_kinect['re_body_angle'],color=cm(norm(data_kinect['re_body_angle'])), units='xy',pivot='middle')
    sns.scatterplot(x="shl_x", y="shl_y", data=body_data,legend='brief', alpha=0.9, color='red',s=100)

    sns.scatterplot(x=data_kinect['shl_x'], y=data_kinect['shl_y'], data=data_kinect, hue='ID_subject', style='ID_subject',legend='brief', alpha=0.5, palette="deep",s=100)


    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')

    return plt


def display_body_direction_stops(body_data, control_points):
    """Display body orientation with stops
    :param body_data: dataframe with trajectory and bodies detected data
    :param control_points: points with real world coordinates in which people stopped
    :return: plot
    """

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    U = np.cos((body_data['re_body_angle']) )
    V = np.sin((body_data['re_body_angle']) )

    sns.set()

    norm = matplotlib.colors.Normalize()
    norm.autoscale(body_data['re_body_angle'])
    cm = matplotlib.cm.viridis

    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots()

    ax.axis(xmin=-3, xmax=3)
    ax.axis(ymin=-0.5, ymax=4.5)

    #sns.scatterplot(x='x', y='y', data=control_points, hue='description', style='description', palette='twilight', s=500, legend=False)
    ax.quiver(body_data['origin_x'], body_data['origin_y'], U, V,scale=0.4, angles=body_data['re_body_angle'],color=cm(norm(body_data['re_body_angle'])), units='xy',pivot='middle', )
    sns.scatterplot(x="origin_x", y="origin_y", data=body_data,legend='brief', alpha=0.9, color='red',s=100)

    fov = Wedge(center=(0, 0), r=4.895, theta1=55, theta2=125, color='purple', alpha=0.05)
    ax.add_artist(fov)

    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')
    plt.title('Standing groups Frontal-Face to Face')

    return plt



def display_body_direction_stops_fov(body_data, control_points, avg_subjects):
    """Display body orientation with stops + field of view

    :param data: dataframe with trajectories and body data
    :param control_points: points with real world coordinates in which people stopped
    :return: plot with data
    """

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    U = np.cos((avg_subjects['re_body_angle']) )
    V = np.sin((avg_subjects['re_body_angle']) )

    sns.set()

    norm = matplotlib.colors.Normalize()
    norm.autoscale(avg_subjects['re_body_angle'])
    cm = matplotlib.cm.viridis

    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots()

    ax.axis(xmin=-3, xmax=3)
    ax.axis(ymin=-0.5, ymax=4.5)

    colors = {1:'red', 2:'green', 0:'dodgerblue', 3:'purple',4:'gray'}

    fov = Wedge(center=(0, 0), r=4.895, theta1=55, theta2=125, color='purple', alpha=0.05)
    ax.add_artist(fov)
    p={}
    i=0
    for index, row in avg_subjects.iterrows():
        p[i]=(row['origin_x'], row['origin_y'])
        #fov = Wedge(center=(row['origin_x'], row['origin_y']), r=1.2, theta1=row['re_body_angle']-30, theta2=row['re_body_angle']+30, color=colors[index[0]], alpha=0.5)
        # fov = Wedge(center=(row['origin_x'], row['origin_y']), r=1.5, theta1=row['re_body_angle']-30, theta2=row['re_body_angle']+30, color=cm(norm(row['re_body_angle'])), alpha=0.5)
        i+=1
        #ax.add_artist(fov)

    print (p)
   
   # circle=draw_o_space(p)
   # ax.add_patch(circle)

    ax.quiver(avg_subjects['origin_x'], avg_subjects['origin_y'], U, V,scale=2, angles=avg_subjects['re_body_angle'],color=cm(norm(avg_subjects['re_body_angle'])), units='xy',pivot='middle', )
    sns.scatterplot(x="origin_x", y="origin_y", data=body_data,legend='brief', alpha=0.9, hue=body_data['shared_stop'] , palette=colors,s=100)

    plt.title('Standing groups. Orientation: Frontal - Diagonal')    
    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')
    #plt.legend(title="Shared stop",fancybox=True)
    cax, _ = mcolorbar.make_axes(plt.gca())
    cb = mcolorbar.ColorbarBase(cax, cmap=matplotlib.cm.viridis, norm=norm)
    cb.set_label('Body Orientation angle')
    

    return plt


def draw_o_space(puntos):
    """Display O space, ellipse between participants

    :param puntos: participants positions
    :return: plot with visualization
    """

    import numpy as np
    
    def define_circle(p1, p2, p3):
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return (None, np.inf)

        # Center of circle
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        return ((cx, cy), radius)

    p1=puntos[0]
    p2=puntos[1]
    p3=puntos[2]
    center, radius = define_circle(p1,p2, p3)
    if center is not None:
        circle = plt.Circle(center, radius, alpha=0.2)
        
        return circle

def display_body_shoulder(data_kinect):
    """ Return a plot with line collections

    :param data: dataframe with trajectories and body data
    :return: plot with data
    """

    fig, ax = plt.subplots()

    segments = list(zip(zip(data_kinect["shr_x"], data_kinect["shr_y"]), zip(data_kinect["shl_x"], data_kinect["shl_y"])))
    lines = []
    
    for item in segments:
        lines.append([item[0], item[1]])

        lc = mc.LineCollection(lines, linewidths=1.0, linestyles='dashed', alpha=0.7)

        ax.add_collection(lc)
        sns.scatterplot(x="shr_x", y="shr_y", data=data_kinect, alpha=0.7,color='red', s=50)
        sns.scatterplot(x="shl_x", y="shl_y", data=data_kinect, alpha=0.7,color='orange', s=50)

    return plt


def display_body_direction(data_kinect, control_points):
    """Display body orientation

    :param data: dataframe with trajectories and body data
    :param control_points: points with real world coordinates in which people stopped
    :return: plot with data
    """

    U = np.cos(data_kinect['re_body_angle'])
    V = np.sin(data_kinect['re_body_angle'])

    colors = np.arctan2(U, V)

    norm = Normalize()
    norm.autoscale(colors)

    colormap = cm.viridis_r

    sns.set()

    fig, ax = plt.subplots()

    lines = []
    lines.append(data_kinect.apply(dp.process_shoulder_segments, axis=1, result_type='expand'))

    lc = mc.LineCollection(lines[0][0], linewidths=1, linestyles='dashed')
    ax.add_collection(lc)


    sns.scatterplot(x="origin_x", y="origin_y", data=data_kinect, hue='ID_exp', style='ID_exp',legend='brief', alpha=0.2, palette="deep",s=100)

    ax.quiver(data_kinect['origin_x'], data_kinect['origin_y'], U, V, color=colormap(norm(colors)), units='x',pivot='tip', width=0.04, scale=2.3 / 0.25, alpha=0.7)

    sns.scatterplot(x='origin_x', y='origin_y', data=control_points, hue='description', style='description',
                    palette='twilight', s=150, legend=False)

    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')

    return plt


def display_body_direction_kinect(data_kinect, control_points):
    """Display body orientation from the depth camera calculated angle

    :param data: dataframe with trajectories and body data
    :param control_points: points with real world coordinates in which people stopped
    :return: plot with data
    """

    U = np.cos((data_kinect['re_body_angle']) * np.pi / 180)
    V = np.sin((data_kinect['re_body_angle']) * np.pi / 180)

    colors = np.arctan2(U, V)

    norm = Normalize()
    norm.autoscale(colors)

    colormap = cm.viridis_r

    sns.set()

    fig, ax = plt.subplots()
    lines = []
    lines.append(data_kinect.apply(dp.process_shoulder_segments, axis=1, result_type='expand'))

    lc = mc.LineCollection(lines[0][0], linewidths=1, linestyles='dashed')
    ax.add_collection(lc)

    sns.scatterplot(x="origin_x", y="origin_y", data=data_kinect, hue='ID_exp', style='ID_exp',legend=False, alpha=0.2, palette="deep",s=100)

    ax.quiver(data_kinect['origin_x'], data_kinect['origin_y'], U, V, color=colormap(norm(colors)), units='x',
              pivot='tip', width=0.04, scale=4 / 0.25, alpha=0.7)

    sns.scatterplot(x='origin_x', y='origin_y', data=control_points, hue='description', style='description',
                    palette='twilight', s=200, legend='brief')

    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')

    plt.title('Body angle Cosh')

    return plt

#ValueError: 'derdgt' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'


def display_aniTrajectory(data_kinect, control_points, stops):
    """Display body orientation animated in plot

    :param data: dataframe with trajectories and body data.
    :param control_points: points with real world coordinates in which people stopped
    :return: plot with data
    """
    data_kinect = data_kinect.sort_values(by=['date_exp'], ascending=[True], ignore_index=True)
    xdata = []
    ydata = []

    xstop=[]
    ystop=[]

    U = np.cos((data_kinect['re_body_angle']) * np.pi / 180)
    V = np.sin((data_kinect['re_body_angle']) * np.pi / 180)

    # U = np.cos((data_kinect['origin_bangle']) * np.pi / 180)
    # V = np.sin((data_kinect['origin_bangle']) * np.pi / 180)

    colors = np.arctan2(U, V)

    norm = Normalize()
    norm.autoscale(colors)

    #colormap = cm.viridis
    colormap = cm.inferno_r

    def update_quiver(i, ax, fig):
        u = np.cos((data_kinect.iloc[i]['origin_bangle']) * np.pi / 180)
        v = np.sin((data_kinect.iloc[i]['origin_bangle']) * np.pi / 180)

        x = data_kinect.iloc[i]['origin_x']
        y = data_kinect.iloc[i]['origin_y']

        Q = ax.quiver(x, y, u, v, color=colormap(norm(colors)), units='x', pivot='tip', width=0.06, scale=2.3 / 0.30, alpha=0.7)

        xdata.append(x)
        ydata.append(y)

        if (x in list(stops['origin_x'])):
            xstop.append(x)
            ystop.append(y)
            line2.set_data(xstop,ystop)

        #line.set_data(xdata, ydata)

        return Q,

    def init_quiver():
        line.set_data([],[])
        return line,


    fig = plt.figure()
    ax = fig.gca()
    line, = ax.plot([], [],lw=2,alpha=0.5)
    line2, = ax.plot([], [], 'ro', markersize=8)
    ax.axis(xmin=-8, xmax=3)
    ax.axis(ymin=-1, ymax=5)
    sns.set()
    sns.scatterplot(x='origin_x', y='origin_y', data=control_points, hue='description', style='description',
                    palette='twilight', s=150, legend=True)

    anim = ani.FuncAnimation(fig, update_quiver,
                                  frames=len(data_kinect),
                                  init_func=init_quiver,
                                  interval=120, fargs=(ax, fig), repeat=False)

    #anim.save('basic_animation.mp4')

    s = anim.to_jshtml()
    with open(data_kinect['ID_exp'][0]+'_1149.html', "w") as f:
        f.write(s)

    plt.show()

    print("done")

    return plt


def display_aniTrajectory_simple(data_kinect, control_points):
    """Display body orientation animated in plot with less info

    :param data: dataframe with trajectories and body data
    :param control_points: points with real world coordinates in which people stopped
    :return: plot with data
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    data_kinect = data_kinect.sort_values(by=['date_exp'], ascending=[True], ignore_index=True)
    xdata = []
    ydata = []

    U = np.cos((data_kinect['re_body_angle']) )
    V = np.sin((data_kinect['re_body_angle']) )

    norm = matplotlib.colors.Normalize()
    norm.autoscale(data_kinect['re_body_angle'])
    cm = matplotlib.cm.viridis

    def update_quiver(i, ax, fig):
        u = np.cos((data_kinect.iloc[i]['re_body_angle']))
        v = np.sin((data_kinect.iloc[i]['re_body_angle']))

        x = data_kinect.iloc[i]['origin_x']
        y = data_kinect.iloc[i]['origin_y']

        angle=data_kinect.iloc[i]['re_body_angle']
        color=cm(norm(data_kinect.iloc[i]['re_body_angle']))

        return ax.quiver(x, y, u, v,color=color, angles=angle, units='xy', pivot='middle', alpha=0.7)

    def init_quiver():
        line.set_data([],[])
        return line,


    fig = plt.figure()
    ax = fig.gca()
    line, = ax.plot([], [],lw=2,alpha=0.5)
    ax.axis(xmin=-2, xmax=4.5)
    ax.axis(ymin=-1, ymax=4.5)
    sns.set()


    # anim = player.Player(fig, update_quiver, frames=len(data_kinect)+1,
    #                               init_func=init_quiver,
    #                               interval=60, fargs=(ax, fig), blit=True)


    anim = ani.FuncAnimation(fig, update_quiver,
                                  frames=len(data_kinect),
                                  init_func=init_quiver,
                                  interval=120, fargs=(ax, fig), repeat=False)

    sns.scatterplot(x='x', y='y', data=control_points, hue='description', style='description',
                    s=150, legend=True)
    # anim.save('basic_animation.mp4')
    s = anim.to_jshtml()
    with open(data_kinect['ID_exp'][0]+'_three1.html', "w") as f:
        f.write(s)

    #plt.show()
    return plt


def angle_cloud(data_kinect, shade_color, title, type):
    """Display body orientations with color range

    :param data: dataframe with trajectories and body data
    :return: plot with data
    """

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    U = np.cos((data_kinect['re_body_angle']) )
    V = np.sin((data_kinect['re_body_angle']) )

    sns.set()

    norm = matplotlib.colors.Normalize()
    norm.autoscale(data_kinect['re_body_angle'])
    cm = matplotlib.cm.viridis

    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots()

    if type==' outliers':

        segments = list(zip(zip(data_kinect["shr_x"], data_kinect["shr_y"]), zip(data_kinect["shl_x"], data_kinect["shl_y"])))
        lines = []
        for item in segments:
            lines.append([item[0], item[1]])

        lc = mc.LineCollection(lines, linewidths=1.0, linestyles='dashed', alpha=0.7)

        ax.add_collection(lc)
        sns.scatterplot(x="shr_x", y="shr_y", data=data_kinect, alpha=0.7,color='red', s=50)
        sns.scatterplot(x="shl_x", y="shl_y", data=data_kinect, alpha=0.7,color='orange', s=50)
       #sns.scatterplot(x="sp_x", y="sp_y", data=data_kinect, alpha=0.7, color='blue', s=50)

    ax.quiver(data_kinect['shl_x'], data_kinect['shl_y'], U, V, angles=data_kinect['re_body_angle'],
              color=cm(norm(data_kinect['re_body_angle'])), units='xy',pivot='middle')

    ax.axis(xmin=-3, xmax=3)
    ax.axis(ymin=-0.5, ymax=4.5)

    fov = Wedge(center=(0, 0), r=4.895, theta1=55, theta2=125, color=shade_color, alpha=0.05)
    ax.add_artist(fov)

    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')
    plt.title('Body Orientation' +type+': '+title+'\n Accepted angle range [-157.5°, -112.5°)')

    cax, _ = mcolorbar.make_axes(plt.gca())
    cb = mcolorbar.ColorbarBase(cax, cmap=matplotlib.cm.viridis, norm=norm)
    cb.set_label('Body Orientation angle')

    return plt


def error_angle_cloud(df_xyz, title, type):
    """Display error in body orientation 

    :param data:
    :return:
    """
    import pandas as pd
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt

    x = df_xyz.iloc[:, -7].values
    y = df_xyz.iloc[:, -6].values
    #z = df_xyz.iloc[:, -8].values
    z = df_xyz.iloc[:, -5].values

    def plot_contour(x, y, z, resolution=100, contour_method='linear'):
        resolution = str(resolution) + 'j'
        X, Y = np.mgrid[min(x):max(x):complex(resolution), min(y):max(y):complex(resolution)]
        points = [[a, b] for a, b in zip(x, y)]
        Z = griddata(points, z, (X, Y), method=contour_method)
        return X, Y, Z

    X, Y, Z = plot_contour(x, y, z, resolution=200, contour_method='linear')


    fig, ax = plt.subplots()


    cs=ax.contourf(X, Y, Z,cmap="viridis_r")
    cb=fig.colorbar(cs, ax=ax, shrink=0.9)
    cb.set_label('Body Orientation angle error')
    #ax.scatter(x, y, color="red", linewidth=1, edgecolor="ivory", s=50)
    plt.ylabel('(Y) Distance from Origin')
    plt.xlabel('(X) Distance from Origin')
    plt.title('Body Orientation error (pair) '+type+': '+title)

    plt.show()


def data_visualization(data,title):
    """ Display statistics

    :param data: dataframe with trajectories and body data
    :return: plot with data
    """

    # Get the label column
    label = data['re_body_angle']

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize=(9, 12))

    # Plot the histogram
    ax[0].hist(label, bins=100)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    l1=ax[0].axvline(label.min(), color='gray', linestyle='dashed', linewidth=2)
    l2=ax[0].axvline(label.mode()[0], color='yellow', linestyle='dashed', linewidth=2)
    l3=ax[0].axvline(label.max(), color='gray', linestyle='dashed', linewidth=2)
    l4=ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
    l5=ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

    # Plot the boxplot
    ax[1].boxplot(label, vert=False)
    ax[1].set_xlabel('Body Angle')
    ax[0].legend((l1, l2, l3,l4,l5), ('Min', 'Mode', 'Max','Mean','Median'),bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Add a title to the Figure
    fig.suptitle('Body Angle Distribution (pair): '+title)

    # Show the figure
    plt.show()