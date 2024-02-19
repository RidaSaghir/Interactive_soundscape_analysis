import json
import logging
import os

import gradio as gr

from .utils import compute_indices, summarise_dataset, list_datasets, update_last_dataset
from graph_plotting import whole_year_plot
from clustering import ClusteringVisualizer

logger = logging.getLogger(__name__)
# Construct the absolute path to the config.json file
config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_file_path) as config_file:
    config = json.load(config_file)

PATH_DATA = config.get('PATH_DATA')
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

clustering = ClusteringVisualizer()


class FrontEndLite:
    def __init__(self, server_name=None, server_port=None):
        self.server_name = server_name or 'localhost'
        self.server_port = server_port or 7860

    def recompute(self, text):
        if text.endswith('?') or 'True' in text:
            return gr.Button(label = 'Recompute', value='Recompute indices?')
        else:
            return gr.Button(label = 'Compute', value='Compute indices')

    def note_msg(self, radio_x_axis):
        if radio_x_axis == 'diel cycle':
            return gr.Text(visible= True)
        else:
            return gr.Text(visible=False)

    def clustering_options(self, selected_option):
        if selected_option == 'pca':
            return gr.Radio(visible=False), gr.Slider(visible=True)
        if selected_option == 'acoustic':
            return gr.Radio(visible=True), gr.Slider(visible=False)

    def ideal_clustering(self, clusters_ideal):
        if clusters_ideal == 'Get optimum number of clusters':
            return gr.Slider(visible=False), gr.Image(visible=True), gr.Textbox(visible=True)
        else:
            return gr.Slider(visible=True), gr.Image(visible=False), gr.Textbox(visible=False)

    def _build_app(self):
        css = """
          #mkd {
            height: 500px; 
            overflow: auto; 
            border: 1px solid #ccc; 
          }
        """

        with gr.Blocks(css=css) as demo:
            gr.HTML(
                """
                    <h1 id="title">Long time scale PAM analyser</h1>
                    <h3>This software provides visualisation of acoustic indices of long time scale acoustic data and computes clusters
                    using unsupervised learning.</h3>
                    <br>
                    <strong>Note: This is a demo software and is undergoing frequent changes.</strong>
                """
            )

            with gr.Tab('Dataset'):
                with gr.Column():
                    with gr.Row():
                        ds_choices, ds_value = list_datasets()
                        dd_ds = gr.Dropdown(
                            choices=ds_choices, value=ds_value, label='Select')
                        dd_ds.change(update_last_dataset, dd_ds)
                        btn_load = gr.Button(value='Summarise', size='sm')
                        btn_compute = gr.Button(label= 'Compute', value='Compute indices', size='sm')
                        txt_out = gr.Textbox(value='Waiting', label='Status')
                        btn_load.click(fn=summarise_dataset, inputs=dd_ds, outputs=txt_out)
                        btn_compute.click(fn=compute_indices, inputs=[dd_ds, btn_compute], outputs=[txt_out])
                        txt_out.change(fn=self.recompute, inputs=[txt_out], outputs=[btn_compute])

            with gr.Tab('Time series'):
                # Add a descriptive text above the button
                # dates = gr.CheckboxGroup(label="Data from following dates were found. Select dates to analyse", choices=detect_datasets())
                with gr.Column():
                    with gr.Row():
                        index_select = gr.Dropdown(
                            [("ACI (Acoustic Complexity Index)", 'ACI'), ("ENT (Temporal Entropy Index)", 'Ht'),
                             ("ENT (Spectral Entropy Index)", 'Hf'),
                             ("CVR LF (Acoustic Cover Index - Low Freq)", 'LFC'),
                             ("CVR MF (Acoustic Cover Index - Mid Freq)", 'MFC'),
                             ("CVR HF (Acoustic Cover Index - High Freq)", 'HFC'),
                             ("EVN (Event Count Index - temp)", 'EVNtCount'),
                             ("EVN (Event Count Index - spec)", 'EVNspCount'),
                             ('EAS (Entropy of Average Spectrum)', 'EAS'),
                             ('ESV (Entropy of Spectral Variance)', 'ESV'),
                             ('ECV (Entropy of Coefficient of Variance)', 'ECV'),
                             ('EPS (Entropy of Spectral Maxima)', 'EPS'),
                             'ACTtCount', 'AGI', 'SNRt'],
                            label="Acoustic Indices")
                        resolution = gr.Radio(["Monthly", "Daily", "Hourly", "Minutes"],
                                              label="Select the resolution for plotting the values")
                    with gr.Row():
                        radio_x_axis = gr.Radio(["year cycle", "diel cycle", "linear"],
                                                label="Select the range for x axis for plotting the values")
                        radio_groupby = gr.Radio(["Year", "Month", "Week", "Day"],
                                                 label="Select the grouping (hue) for plotting the values")
                    with gr.Row():
                        note = gr.Text(
                            value="With this option, there could be only 2 resolutions possible i.e 'Hourly' and 'Minutes'",
                            label="Note:", visible=False)
                        radio_x_axis.change(self.note_msg, inputs=radio_x_axis, outputs=note)

                    with gr.Row():
                        with gr.Column():
                            btn_whole_plot = gr.Button("Plot results")
                            whole_plot = gr.Plot(label="Average ACI over whole timeline")
                            btn_whole_plot.click(whole_year_plot, inputs= [dd_ds, radio_x_axis, radio_groupby, index_select, resolution], outputs=[whole_plot])

            with gr.Tab('Clustering'):
                with gr.Accordion('Clustering based on Acoustic Indices', open=False):
                    with gr.Accordion('K Means', open=False):
                        with gr.Column():
                            with gr.Row():
                                clustering_param = gr.Radio([('Use acoustic indices directly', 'acoustic'),
                                                       ('Use principal component analysis on acoustic indices', 'pca')],
                                                       label="How to cluster?")
                                with gr.Column() as output_col:
                                    with gr.Row():
                                        clusters_ideal = gr.Radio(['Choose the number of clusters', 'Get optimum number of clusters'], label="How to chose number of clusters", interactive=True,
                                                                  visible=True)
                                        max_clusters = gr.Textbox(label='Enter the maximum number of clusters to find optimum number of clusters from.', visible=False)
                                    num_clusters = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                                              label="Select the number of clusters", interactive=True, visible=True)
                                    cluster_indices = gr.CheckboxGroup(['ACI', 'Ht', 'EVNtCount', 'ECV', 'EAS', 'LFC', 'HFC', 'MFC', 'Hf', 'ADI', 'AGI', 'BI'], label= 'Choose the parameters for clustering',
                                                                       visible=False)
                                    num_dimensions = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                                              label="Select the number of dimensions for PCA", interactive=True, visible=False)
                            btn_clusters =gr.Button('Plot Clusters', interactive=True)
                            with gr.Row():
                                clusters_pic =  gr.Plot(label="Clusters based on k-means")
                                sil_score = gr.Plot(label="Best number of clusters based on Silhouette Scores", visible=False)

                    with gr.Accordion('Hierarchical Clustering', open=False):
                        with gr.Row():
                            btn_hierarchical_indices = gr.Button("Perform hierarchical clustering", interactive=True)
                        with gr.Row():
                            clusters_pic_hierar = gr.Plot(label="Clusters based on hierarchical clustering")
                    with gr.Accordion('Cluster Occurrences', open=False):
                        with gr.Accordion('Bar Plots', open=False):
                            gr.Markdown(
                                '<span style="color:#575757;font-size:18px">Barplot for cluster occurrences</span>')
                            with gr.Row():
                                which_cluster = gr.Dropdown(choices=['demo value 1', 'demo value 2'], label='Select the cluster',
                                                             interactive=False)
                                cluster_x_axis = gr.Radio(['Year cycle', 'Diel cycle', 'Linear cycle'],
                                                          label='Select the range for x axis')
                                cluster_hue = gr.Radio(['Year', 'Month', 'Day', 'Hour'],
                                                       label='Select the grouping by option')
                            with gr.Row():
                                submit_btn_occurrences = gr.Button("Generate Barplot", interactive=True)
                            with gr.Row():
                                clusters_bar = gr.Plot()
                        with gr.Accordion('24h Rose Plots', open=False):
                            which_cluster_r = gr.Dropdown(choices=['demo value 1', 'demo value 2'], label='Select the cluster',
                                                          interactive=False)
                            cluster_hue_r = gr.Radio(['Year', 'Month', 'Day'],
                                                     label='Select the grouping by option')
                            with gr.Row():
                                submit_btn_rose = gr.Button("Generate Rose Plot", interactive=True)
                            with gr.Row():
                                clusters_rose = gr.Plot()

                            btn_clusters.click(clustering.kmeans_clustering,
                                               [clustering_param, num_dimensions, cluster_indices, clusters_ideal, num_clusters,
                                                 max_clusters],
                                               outputs=[clusters_pic, sil_score])
                            clustering_param.change(fn=self.clustering_options, inputs=clustering_param, outputs=[cluster_indices, num_dimensions])
                            clusters_ideal.change(fn=self.ideal_clustering, inputs=clusters_ideal,
                                                  outputs=[num_clusters, sil_score, max_clusters])

        self.app = demo

    def launch(self):
        if not hasattr(self, 'app'):
            self._build_app()
        self.app.launch(server_name=self.server_name, server_port=self.server_port)
