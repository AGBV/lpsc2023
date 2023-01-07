import streamlit as st
import numpy as np
import bz2
import _pickle

from plotly.subplots import make_subplots
from plotly import colors
import plotly.graph_objects as go

print('Starting...')

st.set_page_config(
  page_title = 'YASF',
  layout = 'wide'
)

st.title('Yet Another Scattering Framework')

@st.cache
def load_data(path):
  data = bz2.BZ2File(path, 'rb')
  data = _pickle.load(data)
  return data

data = load_data('data/lpsc2023_data_arnaut.pbz2')

wavelength = np.array(data['wavelength']['value'])
scattering_cross_section = data['wavelength']['data']['scattering_cross_section']
extinction_cross_section = data['wavelength']['data']['extinction_cross_section']
single_scattering_albedo = data['wavelength']['data']['single_scattering_albedo']

sampling_points = np.array(data['field']['sampling_points'])
scattered_field = data['field']['scattered_field']

scattering_angles = np.array(data['angle']['value'])
polar_angles      = data['angle']['data']['polar_angles']
azimuthal_angles  = data['angle']['data']['azimuthal_angles']
phase_function    = data['angle']['data']['phase_function']['normal']
phase_function_3d = data['angle']['data']['phase_function']['spatial']
degree_of_linear_polarization    = data['angle']['data']['degree_of_linear_polarization']['normal']
degree_of_linear_polarization_3d = data['angle']['data']['degree_of_linear_polarization']['spatial']
degree_of_linear_polarization_q    = data['angle']['data']['degree_of_linear_polarization_q']['normal']
degree_of_linear_polarization_q_3d = data['angle']['data']['degree_of_linear_polarization_q']['spatial']
degree_of_linear_polarization_u    = data['angle']['data']['degree_of_linear_polarization_u']['normal']
degree_of_linear_polarization_u_3d = data['angle']['data']['degree_of_linear_polarization_u']['spatial']
degree_of_circular_polarization    = data['angle']['data']['degree_of_circular_polarization']['normal']
degree_of_circular_polarization_3d = data['angle']['data']['degree_of_circular_polarization']['spatial']

with st.sidebar:
  wavelength_slider = st.slider('Wavelength Slider', 0, wavelength.size - 1, 0, 1)
  st.write(f'Current wavelength: {wavelength[wavelength_slider]:.2f}nm')
  
  plot_type_options = {
    'Phase Function': dict(
      normal = phase_function,
      three_d = phase_function_3d,
      type = 'log'
    ),
    'Linear Polarization': dict(
      normal = degree_of_linear_polarization,
      three_d = degree_of_linear_polarization_3d,
      type = '-'
    ),
    'Linear Polarization - Q': dict(
      normal = degree_of_linear_polarization_q,
      three_d = np.abs(degree_of_linear_polarization_q_3d),
      type = '-'
    ),
    'Linear Polarization - U': dict(
      normal = degree_of_linear_polarization_u,
      three_d = np.abs(degree_of_linear_polarization_u_3d),
      type = '-'
    ),
    'Circular Polarization': dict(
      normal = degree_of_circular_polarization,
      three_d = np.abs(degree_of_circular_polarization_3d),
      type = '-'
    ),
  }
  plot_type = st.selectbox('Plot Type', plot_type_options.keys(), 0)


with st.container():

  col1, col2 = st.columns(2)
  with col1:
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
    # Scattering Cross-Section
    fig.add_trace(
      go.Scatter(
        x = wavelength / 1e3,
        y = scattering_cross_section,
        name = 'C<sub>sca</sub>'
      ), row=1, col=1
    )
    # Extinction Cross-Section
    fig.add_trace(
      go.Scatter(
        x = wavelength / 1e3,
        y = extinction_cross_section,
        name = 'C<sub>ext</sub>'
      ), row=2, col=1
    )
    # Single-Scattering Albedo
    fig.add_trace(
      go.Scatter(
        x = wavelength / 1e3,
        y = single_scattering_albedo,
        name = 'w'
      ), row=3, col=1
    )

    fig.update_layout(
      title = 'Mixing components',
      height = 900,
      xaxis3 = dict(
        title = 'Wavelength',
        ticksuffix = '&mu;m'
      ),
      yaxis1 = dict(
        title = 'Scattering Cross-section',
        showexponent = 'all',
        exponentformat = 'e'
      ),
      yaxis2 = dict(
        title = 'Extinction Cross-section',
        showexponent = 'all',
        exponentformat = 'e'
      ),
      yaxis3 = dict(
        title = 'Single-Scattering Albedo'
      )
    )
    st.plotly_chart(fig, use_container_width=True)

  with col2:
    eps = np.finfo(float).eps

    vals = np.linalg.norm(np.abs(scattered_field), axis = 2)
    vals_log = np.log(vals+eps)
    vals_log_min = np.min(vals_log)
    vals_log_max = np.max(vals_log)

    tick_vals_log = np.linspace(vals_log_min, vals_log_max, 15)
    tick_vals = [f'{x:.2e}' for x in np.exp(tick_vals_log)-eps]

    fig = go.Figure(
        data = go.Volume(
          x = sampling_points[:, 0].flatten(),
          y = sampling_points[:, 1].flatten(),
          z = sampling_points[:, 2].flatten(),
          value = vals_log[wavelength_slider, :],
          isomin = vals_log_min,
          isomax = vals_log_max,
          opacity = 0.1, # needs to be small to see through all surfaces
          surface_count = 15, # needs to be a large number for good volume rendering
          colorscale = 'jet',
          colorbar = dict(
            tickvals = tick_vals_log,
            ticktext = tick_vals,
          )
      ))
    fig.update_layout(
      height = 800,
      title = 'Electric Field',
    )
    st.plotly_chart(fig, use_container_width=True)

with st.container():

  col1, col2 = st.columns(2)

  with col1:
    fig = make_subplots(rows=2, cols=1,
                        specs = [[{'type': 'xy'}], [{'type': 'polar'}]])
    cmap = colors.sample_colorscale('Jet', np.linspace(0, 1, wavelength.size))
    for wavelength_index in range(wavelength.size):
      fig.add_trace(
        go.Scatter(
          x = scattering_angles * 180 / np.pi,
          y = plot_type_options[plot_type]['normal'][:, wavelength_index],
          line = dict(
            color = cmap[wavelength_index]
          ),
          name = 'Linear Plot',
          legendgrouptitle_text = f'p(θ, {wavelength[wavelength_index]:.2f}nm)',
          legendgroup = f'group{wavelength_index}'
        ), row = 1, col = 1
      )
      fig.add_trace(
        go.Scatterpolar(
          theta = np.concatenate((scattering_angles, 2 * np.pi - np.flip(scattering_angles))) * 180 / np.pi,
          r = np.concatenate((plot_type_options[plot_type]['normal'][:, wavelength_index], np.flip(plot_type_options[plot_type]['normal'][:, wavelength_index]))),
          line = dict(
            color = cmap[wavelength_index]
          ),
          name = 'Polar Plot',
          legendgroup = f'group{wavelength_index}'
        ), row = 2, col = 1
      )
      fig.update_layout(
        height = 1000,
        xaxis1 = dict(
          title = 'Phase Angle',
          ticksuffix = '°',
          tickmode = 'linear',
          tick0 = 0,
          dtick = 45
        ),
        yaxis1 = dict(
          title = plot_type,
          type = plot_type_options[plot_type]['type']
        ),
        polar = dict(
          radialaxis = dict(
            type = plot_type_options[plot_type]['type'],
            dtick = 1
          )
        ),
      )
    st.plotly_chart(fig, use_container_width=True)
  
  with col2:
  
    points = np.vstack([
      np.sin(polar_angles) * np.cos(azimuthal_angles),
      np.sin(polar_angles) * np.sin(azimuthal_angles),
      np.cos(polar_angles)
    ]).T
  
    #%% phase function
    p = np.log(plot_type_options[plot_type]['three_d'] + 1)
    fig = go.Figure(
        go.Scatter3d(
          x = points[:, 0] * p[:, wavelength_slider],
          y = points[:, 1] * p[:, wavelength_slider],
          z = points[:, 2] * p[:, wavelength_slider],
          mode='markers',
          marker=dict(
              size=1,
              color=p[:, wavelength_slider],
              colorscale='Jet',
              opacity=0.8
          )
      )
    )
    p_max = np.max(p, axis = 1)
    points_extrem = points * np.max(p, axis = 1)[:, np.newaxis]
    fig.update_layout(
      height = 800,
      scene = dict(
        xaxis = dict(
          range = [np.min(points_extrem[:, 0]), np.max(points_extrem[:, 0])]
        ),
        yaxis = dict(
          range = [np.min(points_extrem[:, 1]), np.max(points_extrem[:, 1])]
        ),
        zaxis = dict(
          range = [np.min(points_extrem[:, 2]), np.max(points_extrem[:, 2])]
        ),
        aspectratio=dict(
          x=1,
          y=1,
          z=1
        )
      )
    )
    st.plotly_chart(fig, use_container_width=True)