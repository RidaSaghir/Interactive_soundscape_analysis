import gradio as gr
import pandas as pd
from graph_plotting import aci_whole_plot
from graph_plotting import plot_aci_values_regions
from huggingface_hub import login
from clustering import ClusteringVisualizer
from false_color_spec_2 import create_fcs
from correlation_map import create_cormap
from cluster_occurrences import cluster_occurrence
from cluster_rose import rose_plots
from acoustic_region import AcousticRegionAnalyzer
import os


csv_file = 'acoustic_indices.csv'

class AcousticAnalyzerApp:
    def __init__(self):
        self.file_paths_fcs = []
        self.file_paths_interface_fcs = []
        self.df = pd.read_csv(csv_file)
        self.clustering_visualizer = ClusteringVisualizer(self.df)
        self.acoustic_analyzer = AcousticRegionAnalyzer()
        self.data_clustered = []
        self.unique_clusters = []
        self.cluster_options = []

    def calculate_plot_whole_year(self, radio_x_axis, radio_groupby, index_select, resolution):
        if radio_groupby not in self.df:
            raise gr.Error('Cluster information has not been updated yet. Please perform clustering from the tab named "Clustering"')
        avg_aci_whole = aci_whole_plot(self.df, radio_x_axis, radio_groupby, index_select, resolution)
        return avg_aci_whole

    def upload_file_reg(self, file):
        self.file_paths_interface_reg = []
        self.file_paths_reg = []
        #for file in files:
        self.file_paths_interface_reg.append(file.name)
        self.file_paths_reg.append(os.path.basename(file.name))
        return self.file_paths_interface_reg
    def acoustic_reg(self):
        spect, reg = self.acoustic_analyzer.acoustic_regions(self.file_paths_interface_reg)
        regions = []
        for item in reg[0]:
            regions.append(f"Acoustic Region {item}")
        return spect, gr.Radio(choices=regions)

    def acoustic_roi(self, region):
        spect_roi, rois_num = self.acoustic_analyzer.acoustic_reg_roi(region, self.file_paths_interface_reg)
        return spect_roi, gr.Radio(choices=rois_num)

    def roi_audio(self, rois_found):
        roi_audio = self.acoustic_analyzer.roi_play(rois_found)
        return roi_audio

    def call_plot_aci_values_regions(self, plot, hue, region_type):
        acoustic_region_plot = plot_aci_values_regions(self.df, plot, hue, region_type)
        return acoustic_region_plot

    def kmeans_clustering(self, clustering, num_dimensions, clusters_ideal, num_clusters, cluster_indices, max_clusters):
        clusters_pic, self.data_clustered, sil_score = self.clustering_visualizer.kmeans_clustering(clustering, num_dimensions, cluster_indices, clusters_ideal, num_clusters, max_clusters)
        self.unique_clusters = sorted(self.data_clustered['KMeans_Cluster'].unique())
        print(self.unique_clusters)
        cluster_options = [f"Cluster {cluster}" for cluster in self.unique_clusters]
        string_list = [str(element) for element in self.unique_clusters]
        self.cluster_options = list(zip(cluster_options, string_list))
        return clusters_pic, sil_score, gr.Dropdown(choices=self.cluster_options, interactive=True), gr.Dropdown(choices=self.cluster_options, interactive=True)

    def hierar_clustering(self, clustering):
        clusters_pic_hierar = self.clustering_visualizer.hierar_clustering(clustering)

        #clusters_pic = kmeans_clustering(self.df, cluster_indices, clusters_ideal, num_clusters)
        return clusters_pic_hierar

    def cluster_occur(self, which_cluster, cluster_x_axis, cluster_hue):
        clusters_bar = cluster_occurrence(self.data_clustered, which_cluster, cluster_x_axis, cluster_hue)
        return clusters_bar

    def rose_plot(self, which_cluster_r, cluster_hue_r):
        clusters_rose = rose_plots(self.data_clustered, which_cluster_r, cluster_hue_r)
        return clusters_rose
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
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">Event Count Index or EVN</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">An acoustic event corresponds to the period of the signal above a threshold which in this case is 3dBs.</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">EAS</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">Entropy of Average Spectrum.</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">ECV</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">Entropy of Coefficient of Variation (along the time axis for each frequency).</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:18px">EPS</span>')
                gr.Markdown(
                    '<span style="color:#575757;font-size:16px">Entropy of spectral maxima (peaks).</span>')
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
                        radio_groupby = gr.Radio(["Year", "Month", "Week", "Day", ("Cluster", "KMeans_Cluster")],
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
                    with gr.Accordion('Acoustic Region Evaluation', open=False):
                        with gr.Row():
                            upload_region = gr.UploadButton("Click to upload your file", file_types=["audio"], file_count="single")
                            file_output_reg = gr.File(visible=True)
                            submit_reg = gr.Button(label ='Evaluate', interactive=True)
                        with gr.Row():
                            spect = gr.Image(label="Spectrogram")
                            regions_found = gr.Radio([], label= 'Acoustic Regions found in the audio file. Click to evaluate.', interactive=True)
                        with gr.Row():
                            with gr.Column():
                                spect_roi = gr.Image(label="Spectrogram with ROIs")
                            with gr.Column():
                                rois_found = gr.Radio([], label= 'ROIs found in the audio file. Click to listen.', interactive=True)
                                roi_audio = gr.Audio()

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
                                    with gr.Row():
                                        clusters_ideal = gr.Radio(['Choose the number of clusters', 'Get optimum number of clusters'], label="How to chose number of clusters", interactive=True,
                                                                  visible=True)
                                        max_clusters = gr.Textbox(label='Enter the maximum number of clusters to find optimum number of clusters from.', visible=False)
                                    num_clusters = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                                              label="Select the number of clusters", interactive=True, visible=True)
                                    cluster_indices = gr.CheckboxGroup(['ACI', 'ENT', 'EVN', 'ECV', 'EAS', 'LFC', 'HFC', 'MFC', 'EPS'], label= 'Choose the parameters for clustering',
                                                                       visible=False)
                                    num_dimensions = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                                              label="Select the number of dimensions for PCA", interactive=True, visible=False)
                            submit_btn_clusters =gr.Button('Plot Clusters', interactive=True)
                            with gr.Row():
                                clusters_pic =  gr.Plot(label="Clusters based on k-means")
                                sil_score = gr.Plot(label="Best number of clusters based on Silhouette Scores", visible=False)
                    with gr.Accordion('Hierarchical Clustering', open=False):
                        with gr.Row():
                            btn_hierarchical_indices = gr.Button("Perform hierarchical clustering", interactive=True)
                        with gr.Row():
                            #clusters_pic_hierar = gr.Plot(label="Clusters based on hierarchical clustering")
                            clusters_pic_hierar = gr.Image(label="Clusters based on hierarchical clustering")
                    with gr.Accordion('Cluster Occurrences', open=False):
                        with gr.Accordion('Bar Plots', open=False):
                            gr.Markdown(
                                '<span style="color:#575757;font-size:18px">Barplot for cluster occurrences</span>')
                            with gr.Row():
                                which_cluster = gr.Dropdown(choices=self.cluster_options, label='Select the cluster',
                                                            interactive=False)
                                cluster_x_axis = gr.Radio(['Year cycle', 'Diel cycle', 'Linear cycle'], label='Select the range for x axis')
                                cluster_hue = gr.Radio(['Year', 'Month', 'Day', 'Hour'], label= 'Select the grouping by option')
                            with gr.Row():
                                submit_btn_occurrences = gr.Button("Generate Barplot", interactive=True)
                            with gr.Row():
                                clusters_bar = gr.Plot()
                        with gr.Accordion('24h Rose Plots', open=False):
                            which_cluster_r = gr.Dropdown(choices=self.cluster_options, label='Select the cluster',
                                                        interactive=False)
                            cluster_hue_r = gr.Radio(['Year', 'Month', 'Day'],
                                                      label='Select the grouping by option')
                            with gr.Row():
                                submit_btn_rose = gr.Button("Generate Rose Plot", interactive=True)
                            with gr.Row():
                                clusters_rose = gr.Plot()


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
                    return {num_clusters: gr.Slider(visible=False),
                            sil_score: gr.Image(visible=True),
                            max_clusters: gr.Textbox(visible=True)}
                else:
                    return {num_clusters: gr.Slider(visible=True),
                            sil_score: gr.Image(visible=False),
                            max_clusters: gr.Textbox(visible=False)}

            submit_btn.click(self.calculate_plot_whole_year, inputs=[radio_x_axis, radio_groupby, index_select, resolution], outputs=avg_aci_whole)
            upload_region.upload(self.upload_file_reg, upload_region, file_output_reg)
            submit_reg.click(self.acoustic_reg, outputs=[spect, regions_found])
            regions_found.change(self.acoustic_roi, regions_found, outputs=[spect_roi, rois_found])
            rois_found.change(self.roi_audio, rois_found, outputs=[roi_audio])
            submit_btn_2.click(self.call_plot_aci_values_regions, [plot, hue, region_type], acoustic_region_plot)
            submit_btn_clusters.click(self.kmeans_clustering, [clustering, num_dimensions, clusters_ideal, num_clusters, cluster_indices, max_clusters], outputs=[clusters_pic, sil_score, which_cluster, which_cluster_r])
            btn_hierarchical_indices.click(self.hierar_clustering, clustering, outputs=[clusters_pic_hierar])
            submit_btn_occurrences.click(self.cluster_occur, [which_cluster, cluster_x_axis, cluster_hue], clusters_bar)
            submit_btn_rose.click(self.rose_plot, [which_cluster_r, cluster_hue_r], clusters_rose)
            submit_fcs.click(self.call_fcs, inputs=[indices_fcs, unit_fcs], outputs=output_fcs)
            submit_cor.click(self.call_cor, inputs=[threshold_cor], outputs=output_cor)
            radio_x_axis.change(display_options, radio_x_axis, disclaimer)
            clustering.change(clustering_options, clustering, [cluster_indices, num_dimensions])
            clusters_ideal.change(ideal_clustering, clusters_ideal, outputs=[num_clusters, sil_score, max_clusters])
            upload_fcs.upload(self.upload_file_fcs, upload_fcs, file_output_fcs)
            upload_cor.upload(self.upload_file_cor, upload_cor, file_output_cor)


        if __name__ == "__main__":
            #demo.launch(server_name='0.0.0.0', server_port=7860)
            demo.launch()


app = AcousticAnalyzerApp()
app.launch_app()