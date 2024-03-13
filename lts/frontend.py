import json
import logging
import os
import pandas as pd
import gradio as gr

from .utils import compute_indices, summarise_dataset, list_datasets, update_last_dataset
from graph_plotting import whole_year_plot
from clustering import ClusteringVisualizer
from hierar_clustering import hierarchical_clustering
from cluster_occur_bar import cluster_occurrence_bar
from cluster_occur_rose import cluster_occurrence_rose

logger = logging.getLogger(__name__)

config = json.load(open('config.json'))
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

    def clustering_options_rep(self, selected_option):
        if selected_option == 'acoustic':
            return gr.Radio(visible=True)
        else:
            return gr.Radio(visible=False)

    def acoustic_region_display(self, selected_option):
        if selected_option == 'acoustic region':
            return gr.Dropdown(visible=True)
        else:
            return gr.Dropdown(visible=False)

    def clustering_options_dim(self, selected_option):
        if selected_option == 'pca':
            return gr.Slider(visible=True), gr.Radio(visible=False)
        elif selected_option == 'none':
            return gr.Slider(visible=False), gr.Radio(visible=True)

    def ideal_clustering(self, clusters_ideal):
        if clusters_ideal == 'Get optimum number of clusters':
            return gr.Slider(visible=False), gr.Image(visible=True), gr.Textbox(visible=True)
        else:
            return gr.Slider(visible=True), gr.Image(visible=False), gr.Textbox(visible=False)

    def cluster_options_update(self, which_cluster_result):
        print('Hello')
        last_dataset = config["last_dataset"]
        if which_cluster_result == 'pca':
            csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices_pca.csv")
        elif which_cluster_result == 'acoustic':
            csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices.csv")

        data = pd.read_csv(csv_file_path)
        unique_clusters = sorted(data['KMeans_Cluster'].unique())
        cluster_options = [f"Cluster {cluster}" for cluster in unique_clusters]
        return gr.Dropdown(choices=cluster_options, interactive=True)

    def _build_app(self):
        css = """
          #mkd {
            height: 500px; 
            overflow: auto; 
            border: 1px solid #ccc; 
          }
        """

        with gr.Blocks(theme= gr.themes.Default(text_size='lg'), css=css) as demo:
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
                        btn_load = gr.Button(value='Summarise for acoustic indices', size='sm')
                        btn_compute = gr.Button(label= 'Compute', value='Compute indices', size='sm')
                        txt_out = gr.Textbox(value='Waiting', label='Status')
                        btn_load.click(fn=summarise_dataset, inputs=dd_ds, outputs=txt_out)
                        btn_compute.click(fn=compute_indices, inputs=[dd_ds, btn_compute], outputs=[txt_out])
                        txt_out.change(fn=self.recompute, inputs=[txt_out], outputs=[btn_compute])
                    with gr.Row():
                        btn_summarise_vae = gr.Button(value='Summarise for VAE encodings', size='sm')
                        with gr.Column():
                            learning_rate = gr.Textbox(label='Input the learning rate')
                            batch_size = gr.Textbox(label='Input the batch size')
                        with gr.Column():
                            latent_dim = gr.Textbox(label='Input the number of dimensions')
                            btn_compute_vae = gr.Button(label= 'Compute', value='Compute VAE encodings', size='sm')
                        txt_out_vae = gr.Textbox(value='Waiting', label='Status')

            with gr.Tab('Time series'):
                # Add a descriptive text above the button
                with gr.Column():
                    with gr.Row():
                        #TODO: Find out why AGI values are causing problems.
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
                             ('ADI (Acoustic Diversity Index)', 'ADI'),
                             'ACTtCount', 'AGI', 'SNRt', ''],
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
                                with gr.Column():
                                    clustering_rep = gr.Radio([('Acoustic Indices', 'acoustic'),
                                                       ('VAE Encodings', 'vae')],
                                                       label="What representations to use?")
                                    clustering_filter = gr.Radio([('Acoustic Regions', 'acoustic region'),
                                                       ('None', 'none')],
                                                       label="What filters to use?")
                                    clustering_dim_red = gr.Radio([('PCA', 'pca'),
                                                       ('t-SNE', 'tsne'), ('None', 'none')],
                                                       label="What dimensionality reduction technique to use?")
                                with gr.Column():
                                    acoustic_region = gr.Dropdown(['Acoustic Region 1', 'Acoustic Region 2', 'Acoustic Region 3', 'Acoustic Region 4',
                                                                   'Acoustic Region 5', 'Acoustic Region 6', 'Acoustic Region 7', 'Acoustic Region 8',
                                                                   'Acoustic Region 9', 'Acoustic Region 10', 'Acoustic Region 11', 'Acoustic Region 12',
                                                                   'Acoustic Region 13', 'Acoustic Region 14', 'Acoustic Region 15', 'Acoustic Region 16',
                                                                   'Acoustic Region 17', 'Acoustic Region 18', 'Acoustic Region 19', 'Acoustic Region 20'],
                                                                  label="Choose any acoustic region.")
                                    clustering_filter.change(self.acoustic_region_display, inputs=clustering_filter, outputs=acoustic_region)
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
                                with gr.Column():
                                    cluster_playback = gr.Dropdown(choices=[], label="Choose any cluster to play back audios.", interactive=True)
                                    playback_audio = gr.Audio(label="Audio files from the selected cluster")
                            btn_clusters =gr.Button('Plot Clusters', interactive=True)
                            with gr.Row():
                                clusters_pic = gr.Plot(label="Clusters based on k-means")
                                clusters_pic.change(fn=self.cluster_options_update, inputs=[clustering_dim_red],
                                                    outputs=[cluster_playback])
                                sil_score = gr.Plot(label="Best number of clusters based on Silhouette Scores", visible=False)

                    with gr.Accordion('Hierarchical Clustering', open=False):
                        with gr.Row():
                            gr.HTML(
                                """
                                    <strong>Note: The choice of acoustic indices or dimensions of PCA would be taken from the previous calculations from K-Means.</strong>
                                """
                            )
                            num_clusters_hierar = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                                            label="Select the number of clusters", interactive=True,
                                                            visible=True)
                            clustering_param_hierar = gr.Radio([('Use acoustic indices directly', 'acoustic'),
                                                       ('Use principal component analysis on acoustic indices', 'pca')],
                                                       label="How to cluster?")
                            btn_hierarchical = gr.Button("Perform hierarchical clustering", interactive=True)

                        with gr.Row():
                            ward_linkage = gr.Image(label="Ward Linkage Using Eucledean Distance")
                            hierar_clusters = gr.Plot(label="Clusters based on Hierarchical Clustering")
                            btn_hierarchical.click(fn=hierarchical_clustering,
                                                   inputs=[num_clusters_hierar, clustering_param_hierar, cluster_indices], outputs=[ward_linkage, hierar_clusters])
                    with gr.Accordion('Cluster Occurrences', open=False):
                        with gr.Accordion('Bar Plots', open=False):
                            gr.Markdown(
                                '<span style="color:#575757;font-size:18px">Barplot for cluster occurrences</span>')
                            with gr.Row():
                                gr.HTML(
                                    """
                                        <strong>Note: To view the results, you need to perform the clustering first.</strong>
                                    """
                                )
                                which_cluster_result_bar = gr.Radio([('Using PCA', 'pca'), ('None', 'acoustic')],
                                                          label='Which clustering results to use (Dimensionality reduction technique)')
                                which_cluster = gr.Dropdown(choices=['demo value 1', 'demo value 2'], label='Select the cluster',
                                                             interactive=True)
                                cluster_x_axis = gr.Radio(['Year cycle', 'Diel cycle', 'Linear cycle'],
                                                          label='Select the range for x axis')
                                cluster_hue_b = gr.Radio(['year', 'month', 'day', 'hour'],
                                                       label='Select the grouping by option')
                                which_cluster_result_bar.change(fn=self.cluster_options_update, inputs=which_cluster_result_bar, outputs=[which_cluster])

                            with gr.Row():
                                btn_occurrences_bar = gr.Button("Generate Barplot", interactive=True)

                            with gr.Row():
                                clusters_bar = gr.Plot()
                                btn_occurrences_bar.click(fn=cluster_occurrence_bar,
                                                          inputs=[which_cluster, cluster_x_axis, cluster_hue_b, which_cluster_result_bar],
                                                          outputs=clusters_bar)

                        with gr.Accordion('24h Rose Plots', open=False):
                            with gr.Row():
                                which_cluster_result_rose = gr.Radio([('Using PCA', 'pca'), ('None', 'acoustic')],
                                                                    label='Which clustering results to use (Dimensionality reduction technique)')
                                which_cluster_r = gr.Dropdown(choices=['demo value 1', 'demo value 2'], label='Select the cluster',
                                                              interactive=False)
                                cluster_hue_r = gr.Radio(['Year', 'Month', 'Day'],
                                                         label='Select the grouping by option')
                                which_cluster_result_rose.change(fn=self.cluster_options_update, inputs=which_cluster_result_rose, outputs=[which_cluster_r])

                            with gr.Row():
                                btn_occurrences_rose = gr.Button("Generate Rose Plot", interactive=True)
                            with gr.Row():
                                clusters_rose = gr.Plot()
                                btn_occurrences_rose.click(fn=cluster_occurrence_rose,
                                                          inputs=[which_cluster_r, cluster_hue_r,
                                                                  which_cluster_result_rose],
                                                          outputs=clusters_rose)

                            btn_clusters.click(clustering.kmeans_clustering,
                                               [clustering_rep, clustering_filter, acoustic_region, clustering_dim_red, num_dimensions, cluster_indices, clusters_ideal, num_clusters,
                                                 max_clusters],
                                               outputs=[clusters_pic, sil_score])
                            clustering_rep.change(fn=self.clustering_options_rep, inputs=clustering_rep, outputs=[cluster_indices])
                            clustering_dim_red.change(fn=self.clustering_options_dim, inputs=clustering_dim_red, outputs=[num_dimensions, cluster_indices])
                            clusters_ideal.change(fn=self.ideal_clustering, inputs=clusters_ideal,
                                                  outputs=[num_clusters, sil_score, max_clusters])
        # All event listeners

        self.app = demo

    def launch(self):
        if not hasattr(self, 'app'):
            self._build_app()
        self.app.launch(server_name=self.server_name, server_port=self.server_port)
