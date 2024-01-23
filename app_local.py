import gradio as gr
import pandas as pd
from graph_plotting import aci_whole_plot
from graph_plotting import plot_aci_values_regions
from huggingface_hub import login
from clustering import ClusteringVisualizer
from false_color_spec_2 import create_fcs
from correlation_map import create_cormap
import os


csv_file = 'parsed_info.csv'

class AcousticAnalyzerApp:
    def __init__(self):
        self.file_paths_fcs = []
        self.file_paths_interface_fcs = []
        self.df = pd.read_csv(csv_file)
        self.clustering_visualizer = ClusteringVisualizer(self.df)
        self.data_clustered = []
        self.unique_clusters = []
        self.cluster_options = []

    def calculate_plot_whole_year(self, radio_x_axis, radio_groupby, index_select, resolution):
        avg_aci_whole = aci_whole_plot(self.df, radio_x_axis, radio_groupby, index_select, resolution)
        return avg_aci_whole

    def call_plot_aci_values_regions(self, plot, hue, region_type):
        acoustic_region_plot = plot_aci_values_regions(self.df, plot, hue, region_type)
        return acoustic_region_plot

    def kmeans_clustering(self, clustering, num_dimensions, clusters_ideal, num_clusters, cluster_indices):
        clusters_pic, self.data_clustered = self.clustering_visualizer.kmeans_clustering(clustering, num_dimensions, cluster_indices, clusters_ideal, num_clusters)
        self.unique_clusters = sorted(self.data_clustered['KMeans_Cluster'].unique())
        self.cluster_options = [f"Cluster {cluster}" for cluster in self.unique_clusters]
        print(self.cluster_options)

        return clusters_pic, gr.Dropdown(choices=self.cluster_options, interactive=True)

    def hierar_clustering(self, clustering):
        clusters_pic_hierar = self.clustering_visualizer.hierar_clustering(clustering)

        #clusters_pic = kmeans_clustering(self.df, cluster_indices, clusters_ideal, num_clusters)
        return clusters_pic_hierar

    def upload_file_fcs(self, files):
        self.file_paths_interface_fcs = []
        self.file_paths_fcs = []
        for file in files:
            self.file_paths_interface_fcs.append(file.name)
            self.file_paths_fcs.append(os.path.basename(file.name))

        return self.file_paths_interface_fcs

    def upload_file_cor(self, files):
        self.file_paths_interface_cor = []
        self.file_paths_cor = []
        for file in files:
            self.file_paths_interface_cor.append(file.name)
            self.file_paths_cor.append(os.path.basename(file.name))

        return self.file_paths_interface_cor


    def call_fcs(self, indices_fcs, unit_fcs):
        output_fcs = create_fcs(indices_fcs, self.file_paths_interface_fcs, unit_fcs)
        return output_fcs

    def call_cor(self, threshold):
        output_cor = create_cormap(self.file_paths_interface_cor, threshold)
        return output_cor

    def launch_app(self):
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
                    <h1 id="title">Long time scale acoustic analyser</h1>
                    <h3>This software provides visualisation of acoustic indices of long time scale acoustic data and computes clusters
                    using unsupervised learning.</h3>
                    <br>
                    <strong>Note: This is a demo software and is undergoing frequent changes.</strong>
                """
            )
            with gr.Accordion('Description', open=False):
                gr.Markdown('<span style="color:#3B5998;font-size:20px">Acoustic Indices</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">Following are the precise descriptions of the acoustic indices being used.</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">Acoustic Complexity Index or ACI</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">Measures the amplitude oscillation in eah frequency bin.</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">Temporal Entropy Index or ENT</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">It measures the energy dispersion over the frames of each frequency.</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">Acoustic Cover Index or CVR</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">It calculates the portion of noise-reduced spectrogram which surpasses 2dB.'
                    'It is calculated for 3 bandwidths; low frequency band (LF), medium frequency band (MF) and high frequency band (HF).'
                    'By default, the values for these are; LF: 0 - 1000Hz, MF: 1000 - 8000Hz, HF: 8000-11000Hz  </span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">Spectral Activity Index or ACT</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">It corresponds to the portion of the spectrogram above a threshold frequency per frequency along time axis</span>')
            with gr.Tab('Whole Year Plots'):
                    # Add a descriptive text above the button
                    #dates = gr.CheckboxGroup(label="Data from following dates were found. Select dates to analyse", choices=detect_datasets())
                with gr.Column():
                    with gr.Row():
                        index_select = gr.Dropdown(
                    [("ACI (Acoustic Complexity Index)", 'ACI'), ("ENT (Temporal Entropy Index)", 'ENT'), ("CVR LF (Acoustic Cover Index - Low Freq)", 'LFC'),
                     ("CVR MF (Acoustic Cover Index - Mid Freq)", 'MFC'), ("CVR HF (Acoustic Cover Index - High Freq)", 'HFC'),
                     ("EVN (Event Count Index)", 'EVN')],
                            label="Acoustic Indices", info="Will add more indices later!"
                )
                        resolution = gr.Radio(["Monthly", "Daily", "Hourly", "Minutes"],
                                                label="Select the resolution for plotting the values")
                    with gr.Row():
                        radio_x_axis = gr.Radio(["year cycle", "diel cycle", "linear"],
                                                label="Select the range for x axis for plotting the values")
                        radio_groupby = gr.Radio(["Year", "Month", "Week", "Day"],
                                                 label="Select the grouping (hue) for plotting the values")
                    with gr.Row():
                        disclaimer = gr.Text(value="With this option, there could be only 2 resolutions possible i.e 'Hourly' and 'Minutes'", label="Note:", visible=False)
                    with gr.Row():
                        with gr.Column():
                            submit_btn = gr.Button("Plot results")
                            avg_aci_whole = gr.Plot(label="Average ACI over whole timeline")

            with gr.Tab('Acoustic features evaluation'):
                with gr.Column():
                    with gr.Accordion('False Color Spectrogram', open=False):
                        with gr.Row():
                            upload_fcs = gr.UploadButton("Click to Upload Files for false color spectrogram", file_types=["audio"],
                                                            file_count="multiple")
                        with gr.Row():
                            gr.Markdown(
                                '<span style="color:#575757;font-size:16px">Note: Please wait until all the chosen files are uploaded. </span>')
                        with gr.Row():
                            indices_fcs = gr.CheckboxGroup([("ACI", 'ACI_per_bin'), ("ENT", 'Ht_per_bin'), ("EVN", 'EVNspCount_per_bin'),
                                                            ("ACT", 'ACTspCount'), ],
                                                           label="Select three acoustic indices for the spectrogram")
                            unit_fcs = gr.Radio(['minutes', 'hours', 'days', 'weeks'], label="Select the units for spectrogram")
                        with gr.Row():
                            submit_fcs = gr.Button('Create false color spectrogram', interactive=True)
                        with gr.Accordion('Files uploaded', open=False):
                            with gr.Row():
                                file_output_fcs = gr.File(visible=True)
                        with gr.Row():
                            output_fcs = gr.Image(label="False color spectrogram", type="pil")

                    with gr.Accordion('Correlation Map', open=False):
                        with gr.Row():
                            upload_cor = gr.UploadButton("Click to Upload Files for false color spectrogram", file_types=["audio"],
                                                            file_count="multiple")
                        with gr.Row():
                            threshold_cor = gr.Textbox(label="Threshold value. E.g: 0.5")
                            with gr.Column():
                                submit_cor = gr.Button('Create correlation maüp', interactive=True)
                        with gr.Accordion('Files Uploaded', open=False):
                            with gr.Row():
                                file_output_cor = gr.File(visible=True)
                        with gr.Row():
                            # output_cor = gr.Image(label="Correlation Map", type="pil")
                            output_cor = gr.Plot(label="Correlation Map")

                    with gr.Accordion('Acoustic indices plot according to region', open=False):
                        region_type = gr.CheckboxGroup(
                                ["Region 1", "Region 2", "Region 3", "Region 4",
                                 "Region 5", "Region 6", "Region 7", "Region 8",
                                 "Region 9", "Region 10", "Region 11", "Region 12",
                                 "Region 13", "Region 14", "Region 15", "Region 16",
                                 "Region 17", "Region 18", "Region 19", "Region 20"], label="Select acoustic regions to compare")
                        with gr.Row():
                            hue = gr.Radio(["Years on x-axis", "Regions on x-axis"], label='Select the plot settings')
                            plot = gr.Radio(["Bar plot", "Time series plot"], label='Select the type of plot')
                        submit_btn_2 = gr.Button("Plot according to regions")
                        with gr.Column():
                            # Create Gradio blocks for outputs
                            acoustic_region_plot = gr.Plot(label="Average ACI over acoustic regions")

            with gr.Tab('Clustering'):
                with gr.Accordion('Clustering based on Acoustic Indices', open=False):
                    with gr.Accordion('K Means', open=False):
                        with gr.Column():
                            with gr.Row():
                                clustering = gr.Radio([('Use acoustic indices directly', 'acoustic'),
                                                       ('Use principal component analysis on acoustic indices', 'pca')],
                                                       label="How to cluster?")
                                with gr.Column() as output_col:
                                    clusters_ideal = gr.Radio(['Choose own choice of number of clusters', 'Get optimum number of clusters'], label="How to chose number of clusters", interactive=True,
                                                              visible=True)
                                    num_clusters = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                                              label="Select the number of clusters", interactive=True, visible=True)
                                    cluster_indices = gr.CheckboxGroup(['ACI', 'ENT', 'EVN', 'ECV', 'EAS', 'LFC', 'HFC', 'MFC', 'EPS'], label= 'Choose the parameters for clustering',
                                                                       visible=False)
                                    num_dimensions = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                                              label="Select the number of dimensions for PCA", interactive=True, visible=False)
                            submit_btn_clusters =gr.Button('Plot Clusters', interactive=True)
                            clusters_pic =  gr.Plot(label="Clusters based on k-means")
                    with gr.Accordion('Hierarchical Clustering', open=False):
                        with gr.Row():
                            btn_hierarchical_indices = gr.Button("Perform hierarchical clustering", interactive=True)
                        with gr.Row():
                            #clusters_pic_hierar = gr.Plot(label="Clusters based on hierarchical clustering")
                            clusters_pic_hierar = gr.Image(label="Clusters based on hierarchical clustering")
                    with gr.Accordion('Cluster Occurrences (Under development)', open=False):
                        gr.Markdown(
                            '<span style="color:#575757;font-size:18px">Barplot for cluster occurrences</span>')
                        with gr.Row():
                            which_cluster = gr.Dropdown(choices=self.cluster_options, label='Select the cluster',
                                                        interactive=False)
                            cluster_cycle = gr.Radio(['Year', 'Diel', 'Linear'], label='Select the cycle')
                        with gr.Row():
                            submit_btn_occurrences = gr.Button("Generate Barplot", interactive=True)

            def display_options(selected_option):
                if selected_option == 'diel cycle':
                    return gr.Text(visible=True)
                else :
                    return gr.Text(visible=False)

            def clustering_options(selected_option):
                if selected_option == 'pca':
                    return {cluster_indices: gr.Radio(visible=False),
                            num_dimensions: gr.Slider(visible=True)}
                if selected_option == 'acoustic':
                    return {cluster_indices: gr.Radio(visible=True),
                            num_dimensions: gr.Slider(visible=False)}

            def ideal_clustering(clusters_ideal):
                if clusters_ideal == 'Get optimum number of clusters':
                    return gr.Slider(visible=False)
                else:
                    return gr.Slider(visible=True)

            submit_btn.click(self.calculate_plot_whole_year, inputs=[radio_x_axis, radio_groupby, index_select, resolution], outputs=avg_aci_whole)
            submit_btn_2.click(self.call_plot_aci_values_regions, [plot, hue, region_type], acoustic_region_plot)
            submit_btn_clusters.click(self.kmeans_clustering, [clustering, num_dimensions, clusters_ideal, num_clusters, cluster_indices], outputs=[clusters_pic, which_cluster])
            btn_hierarchical_indices.click(self.hierar_clustering, clustering, outputs=[clusters_pic_hierar])
            submit_fcs.click(self.call_fcs, inputs=[indices_fcs, unit_fcs], outputs=output_fcs)
            submit_cor.click(self.call_cor, inputs=[threshold_cor], outputs=output_cor)
            radio_x_axis.change(display_options, radio_x_axis, disclaimer)
            clustering.change(clustering_options, clustering, [cluster_indices, num_dimensions])
            clusters_ideal.change(ideal_clustering, clusters_ideal, num_clusters)
            upload_fcs.upload(self.upload_file_fcs, upload_fcs, file_output_fcs)
            upload_cor.upload(self.upload_file_cor, upload_cor, file_output_cor)


        if __name__ == "__main__":
            demo.launch()

app = AcousticAnalyzerApp()
app.launch_app()