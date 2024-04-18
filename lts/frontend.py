import json
import logging
import os
import pandas as pd
import gradio as gr
import random
import soundfile as sf

from .utils import (compute_indices, summarise_dataset, list_datasets, update_last_dataset,
                    update_clustering_rep, update_clustering_mode, update_clustering_filter, update_dim_red, update_region)
from graph_plotting import whole_year_plot
from bandpass_audio_retrieval import filter_audio
from clustering import ClusteringVisualizer
from hierar_clustering import hierarchical_clustering
from multiple_cluster_occur import cluster_occurrence_multi
from cluster_occur_bar import cluster_occurrence_bar
from cluster_occur_rose import cluster_occurrence_rose

logger = logging.getLogger(__name__)

# config = json.load(open('config.json'))
# PATH_DATA = config.get('PATH_DATA')
# PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

PATH_DATA = None
last_dataset = None
PATH_EXP = None
clustering = ClusteringVisualizer()

class FrontEndLite:
    def __init__(self, server_name=None, server_port=None):
        self.server_name = server_name or 'localhost'
        self.server_port = server_port or 7860

    def load_config(self):
        self.config = json.load(open('config.json'))
        self.path_data = self.config["PATH_DATA"]
        self.last_dataset = self.config["last_dataset"]
        self.path_exp = os.path.join(os.path.dirname(self.path_data), 'exp')
        self.clustering_rep = self.config["clustering_rep"]
        self.resolution_features = self.config["resolution"]
        self.clustering_mode = self.config["clustering_mode"]
        self.dim_red_mode = self.config["dim_red_mode"]
        self.clustering_filter = self.config["clustering_filter"]
        self.acoustic_region = self.config["acoustic_region"]

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

    def num_dimension_components(self, selected_option):
        if selected_option in ['pca', 'tsne', 'umap', 'isomap']:
            return gr.Slider(visible=True), gr.Radio(visible=False)
        elif selected_option == 'none':
            return gr.Slider(visible=False), gr.Radio(visible=True)

    def ideal_clustering(self, clusters_ideal):
        if clusters_ideal == 'Get optimum number of clusters':
            return gr.Slider(visible=False), gr.Image(visible=True), gr.Textbox(visible=True)
        else:
            return gr.Slider(visible=True), gr.Image(visible=False), gr.Textbox(visible=False)

    def get_clusters(self):
        self.load_config()
        csv = f'{self.clustering_rep}_{self.resolution_features}_{self.clustering_mode}_{self.dim_red_mode}_{self.clustering_filter}_{self.acoustic_region}.csv'
        self.audio_file_path = os.path.join(self.path_exp, self.last_dataset, csv)
        if os.path.exists(self.audio_file_path):
            df = pd.read_csv(self.audio_file_path)
            column = f'{self.clustering_mode} labels'
            unique_clusters = sorted(df[column].unique())
            cluster_options = [f"Cluster {cluster}" for cluster in unique_clusters]
            return (gr.Dropdown(choices=cluster_options, interactive=True),
                    gr.Dropdown(choices=cluster_options, interactive=True),
                    gr.Dropdown(choices=cluster_options, interactive=True))

    def retrieve_audios(self, chosen_cluster):
        self.load_config()
        df = pd.read_csv(self.audio_file_path, index_col=0)
        cluster_number = int(chosen_cluster.split()[1])
        cluster_title = f'{self.clustering_mode} labels'
        filtered_df = df[df[cluster_title] == cluster_number]
        files = []
        for row in filtered_df['File Name']:
            files.append(os.path.join(self.path_data, self.last_dataset, row))
        output_files = random.sample(files, min(5, len(files)))
        # Bandpass filtering
        if self.acoustic_region != 'none':
            output_files = filter_audio(output_files, self.acoustic_region)
            wav_file_paths = []
            for i, (y, sr) in enumerate(output_files):
                file_path = f"filtered_output_{i}.wav"
                sf.write(file_path, y, sr)
                wav_file_paths.append(file_path)
            output_files = wav_file_paths

        return (gr.Audio(value=output_files[0], visible=True), gr.Audio(value=output_files[1], visible=True),
                gr.Audio(value=output_files[2], visible=True), gr.Audio(value=output_files[3], visible=True),
                gr.Audio(value=output_files[4], visible=True))

    def assign_label(self, chosen_cluster, cluster_label):
        df = pd.read_csv(self.audio_file_path, index_col=0)
        cluster_title = f'{self.config["clustering_mode"]} labels'
        cluster_number = int(chosen_cluster.split()[1])
        selected_rows = df[df[cluster_title] == cluster_number]
        print(f'selected rows : {selected_rows.index}')
        df.loc[selected_rows.index, 'assigned_labels'] = cluster_label
        df.to_csv(self.audio_file_path)
        return

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

                        btn_load = gr.Button(value='Summarise for acoustic indices', size='sm')
                        btn_compute = gr.Button(label= 'Compute', value='Compute indices', size='sm')
                        txt_out = gr.Textbox(value='Waiting', label='Status')
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
                    with gr.Row():
                        with gr.Column():
                            btn_whole_plot = gr.Button("Plot results")
                            whole_plot = gr.Plot(label="Average ACI over whole timeline")

            with gr.Tab('Clustering Panel'):
                with gr.Accordion('Perform Clustering', open=False):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                clustering_mode = gr.Radio([('K-Means', 'k_means'),
                                                   ('HDBSCAN', 'hdbscan'), ('DBSCAN', 'dbscan'), ('affinity', 'affinity'), ('spectral', 'spectral')],
                                                   label="What clustering mode to use?")
                                clustering_rep = gr.Radio([('Acoustic Indices', 'acoustic_indices'),
                                                   ('VAE Encodings', 'vae')],
                                                   label="What representations to use?")
                                clustering_filter = gr.Radio([('Acoustic Regions', 'acoustic region'),
                                                   ('None', 'none')],
                                                   label="What filters to use?")
                                dim_red = gr.Radio([('PCA', 'pca'),
                                                   ('t-SNE', 'tsne'), ('UMAP', 'umap'), ('Isomap', 'isomap'), ('None', 'none')],
                                                   label="What dimensionality reduction technique to use?")

                            with gr.Column():
                                acoustic_region = gr.Dropdown(['Acoustic Region 1', 'Acoustic Region 2', 'Acoustic Region 3', 'Acoustic Region 4',
                                                               'Acoustic Region 5', 'Acoustic Region 6', 'Acoustic Region 7', 'Acoustic Region 8',
                                                               'Acoustic Region 9', 'Acoustic Region 10', 'Acoustic Region 11', 'Acoustic Region 12',
                                                               'Acoustic Region 13', 'Acoustic Region 14', 'Acoustic Region 15', 'Acoustic Region 16',
                                                               'Acoustic Region 17', 'Acoustic Region 18', 'Acoustic Region 19', 'Acoustic Region 20'],
                                                              label="Choose any acoustic region.")
                                with gr.Row():
                                    clusters_ideal = gr.Radio(['Choose the number of clusters', 'Get optimum number of clusters'], label="How to chose number of clusters", interactive=True,
                                                              visible=True)
                                    max_clusters = gr.Textbox(label='Enter the maximum number of clusters to find optimum number of clusters from.', visible=False)
                                num_clusters = gr.Slider(minimum=2, maximum=10, value=2, step=1,
                                                          label="Select the number of clusters", interactive=True, visible=True)
                                chosen_indices = gr.CheckboxGroup(['ACI', 'Ht', 'EVNtCount', 'ECV', 'EAS', 'LFC', 'HFC', 'MFC', 'Hf', 'ADI', 'BI'], label= 'Choose the parameters for clustering',
                                                                   visible=False)
                                num_dimensions = gr.Slider(minimum=3, maximum=10, value=3, step=1,
                                                          label="Select the number of dimensions for selected DR method", interactive=True, visible=False)
                            with gr.Column():
                                cluster_playback = gr.Dropdown(choices=[], label="Choose any cluster to play back audios.", interactive=True, allow_custom_value=True)
                                retrieve_clusters = gr.Button("Get cluster audios")

                        btn_clusters =gr.Button('Plot Clusters', interactive=True)
                        with gr.Row():
                            clusters_pic = gr.Plot(label="Clusters based on chosen method")
                            with gr.Column():
                                playback_audio_1 = gr.Audio(label="Audio 1", visible=False)
                                playback_audio_2 = gr.Audio(label="Audio 2", visible=False)
                                playback_audio_3 = gr.Audio(label="Audio 3", visible=False)
                                playback_audio_4 = gr.Audio(label="Audio 4", visible=False)
                                playback_audio_5 = gr.Audio(label="Audio 5", visible=False)
                                cluster_label = gr.Textbox(label="Assign label to cluster")
                                submit_label = gr.Button("Submit cluster label")
                            sil_score = gr.Plot(label="Best number of clusters based on Silhouette Scores", visible=False)

                    # with gr.Accordion('Hierarchical Clustering', open=False):
                    #     with gr.Row():
                    #         gr.HTML(
                    #             """
                    #                 <strong>Note: The choice of acoustic indices or dimensions of PCA would be taken from the previous calculations from K-Means.</strong>
                    #             """
                    #         )
                    #         num_clusters_hierar = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                    #                                         label="Select the number of clusters", interactive=True,
                    #                                         visible=True)
                    #         clustering_param_hierar = gr.Radio([('Use acoustic indices directly', 'acoustic'),
                    #                                    ('Use principal component analysis on acoustic indices', 'pca')],
                    #                                    label="How to cluster?")
                    #         btn_hierarchical = gr.Button("Perform hierarchical clustering", interactive=True)
                    #
                    #     with gr.Row():
                    #         ward_linkage = gr.Image(label="Ward Linkage Using Eucledean Distance")
                    #         hierar_clusters = gr.Plot(label="Clusters based on Hierarchical Clustering")
                    #         btn_hierarchical.click(fn=hierarchical_clustering,
                    #                                inputs=[num_clusters_hierar, clustering_param_hierar, chosen_indices], outputs=[ward_linkage, hierar_clusters])
                with gr.Accordion('Cluster Occurrences', open=False):
                    with gr.Accordion('Multiple Cluster Plots', open=False):
                        with gr.Row():
                            x_average = gr.Radio([('By month', 'month'), ('By date', 'date'), ('By hour', 'hour')],
                                                  label='Select the averaging method')
                        with gr.Row():
                            btn_occurrences_multi = gr.Button("Generate plot", interactive=True)
                        with gr.Row():
                            clusters_multi = gr.Plot()
                    with gr.Accordion('Single Cluster Plots', open=False):
                        gr.Markdown(
                            '<span style="color:#575757;font-size:18px">Barplot for cluster occurrences</span>')
                        with gr.Row():
                            gr.HTML(
                                """
                                    <strong>Note: To view the results, you need to perform the clustering first.</strong>
                                """
                            )
                            which_cluster = gr.Dropdown(choices=['demo value 1', 'demo value 2'], label='Select the cluster',
                                                         interactive=True)
                            cluster_x_axis = gr.Radio(['Year cycle', 'Diel cycle', 'Linear cycle'],
                                                      label='Select the range for x axis')
                            cluster_hue_b = gr.Radio(['year', 'month', 'day', 'hour'],
                                                   label='Select the grouping by option')
                        with gr.Row():
                            btn_occurrences_bar = gr.Button("Generate Barplot", interactive=True)

                        with gr.Row():
                            clusters_bar = gr.Plot()


                    with gr.Accordion('24h Rose Plots', open=False):
                        with gr.Row():
                            which_cluster_r = gr.Dropdown(choices=['demo value 1', 'demo value 2'], label='Select the cluster',
                                                          interactive=False)
                            cluster_hue_r = gr.Radio(['Year', 'Month', 'Day'],
                                                     label='Select the grouping by option')
                        with gr.Row():
                            btn_occurrences_rose = gr.Button("Generate Rose Plot", interactive=True)
                        with gr.Row():
                            clusters_rose = gr.Plot()

            # ALL EVENT LISTENERS

            # For data set
            dd_ds.change(update_last_dataset, dd_ds)
            btn_load.click(fn=summarise_dataset, inputs=dd_ds, outputs=txt_out)
            btn_compute.click(fn=compute_indices, inputs=[dd_ds, btn_compute], outputs=[txt_out])
            txt_out.change(fn=self.recompute, inputs=[txt_out], outputs=[btn_compute])

            # For time series plot
            radio_x_axis.change(self.note_msg, inputs=radio_x_axis, outputs=note)
            btn_whole_plot.click(whole_year_plot, inputs=[radio_x_axis, radio_groupby, index_select, resolution],
                                 outputs=[whole_plot])


            # For clustering
            clustering_mode.change(fn=update_clustering_mode, inputs=[clustering_mode])
            clustering_filter.change(self.acoustic_region_display, inputs=clustering_filter, outputs=acoustic_region)
            clusters_ideal.change(fn=self.ideal_clustering, inputs=clusters_ideal,
                                  outputs=[num_clusters, sil_score, max_clusters])
            clustering_rep.change(fn=update_clustering_rep, inputs=[clustering_rep])
            clustering_rep.change(fn=self.clustering_options_rep, inputs=clustering_rep, outputs=[chosen_indices])
            clustering_filter.change(fn=update_clustering_filter, inputs=[clustering_filter])
            dim_red.change(fn=update_dim_red, inputs=[dim_red])
            acoustic_region.change(fn=update_region, inputs=[acoustic_region])
            dim_red.change(fn=self.num_dimension_components, inputs=dim_red, outputs=[num_dimensions, chosen_indices])
            btn_clusters.click(clustering.clustering,
                               [num_dimensions, chosen_indices, clusters_ideal, num_clusters,
                                max_clusters],
                               outputs=[clusters_pic, sil_score,])
            retrieve_clusters.click(fn=self.get_clusters, outputs=[cluster_playback, which_cluster, which_cluster_r])
            cluster_playback.change(fn=self.retrieve_audios, inputs=[cluster_playback], outputs=[playback_audio_1, playback_audio_2,
                                                                                                 playback_audio_3, playback_audio_4, playback_audio_5])
            submit_label.click(fn=self.assign_label, inputs=[cluster_playback, cluster_label])


            # For cluster occurrences
            btn_occurrences_multi.click(fn=cluster_occurrence_multi,
                                      inputs=[x_average],
                                      outputs=clusters_multi)
            btn_occurrences_bar.click(fn=cluster_occurrence_bar,
                                      inputs=[which_cluster, cluster_x_axis, cluster_hue_b],
                                      outputs=clusters_bar)
            btn_occurrences_rose.click(fn=cluster_occurrence_rose,
                                       inputs=[which_cluster_r, cluster_hue_r],
                                       outputs=clusters_rose)

        self.app = demo

    def launch(self):
        if not hasattr(self, 'app'):
            self._build_app()
        self.app.launch(server_name=self.server_name, server_port=self.server_port, share=True)
