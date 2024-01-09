import gradio as gr
import pandas as pd
from graph_plotting import aci_whole_plot
from graph_plotting import plot_aci_values_regions
from huggingface_hub import login
from clustering import kmeans_clustering

access_token_write = "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"
login(token = access_token_write)

csv_file = 'parsed_info.csv'
df = pd.read_csv(csv_file)

unique_years = df['Year'].astype(str).str.zfill(2) + '/' + df['Month'].astype(str).str.zfill(2) + '/' + df[
    'Day'].astype(str).str.zfill(2)
unique_years = unique_years.drop_duplicates().tolist()
# unique_years = df['Year'].unique()
def calculate_plot_whole_year(radio_x_axis, radio_groupby, index_select, resolution):
    avg_aci_whole = aci_whole_plot(df, radio_x_axis, radio_groupby, index_select, resolution)
    return avg_aci_whole

def call_plot_aci_values_regions(plot, hue, region_type):
    acoustic_region_plot = plot_aci_values_regions(df, plot, hue, region_type)
    return acoustic_region_plot

def clustering(clusters_ideal, num_clusters, cluster_indices):
    clusters_pic = kmeans_clustering(df, cluster_indices, clusters_ideal, num_clusters)
    return clusters_pic
def detect_datasets():
    unique_years = df['Year'].astype(str).str.zfill(2) + '/' + df['Month'].astype(str).str.zfill(2) + '/' + df[
        'Day'].astype(str).str.zfill(2)
    unique_years = unique_years.drop_duplicates().tolist()
    #unique_years = df['Year'].unique()
    return unique_years

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
    with gr.Tab('Whole Year Plots'):
        with gr.Column():
            # Add a descriptive text above the button
            gr.Label("Acoustic Region Analyzer")
            #dates = gr.CheckboxGroup(label="Data from following dates were found. Select dates to analyse", choices=detect_datasets())

        with gr.Column():
            with gr.Row():
                # Look into (name, value) thing of drop down option gradio
                index_select = gr.Dropdown(
            ["ACI (Acoustic Complexity Index)", "ENT (Temporal Entropy Index)", "CVR LF (Acoustic Cover Index - Low Freq)",
             "CVR MF (Acoustic Cover Index - Mid Freq)", "CVR HF (Acoustic Cover Index - High Freq)",
             "EVN (Event Count Index)"],
                    label="Acoustic Indices", info="Will add more indices later!"
        )
                resolution = gr.Radio(["Monthly", "Weekly", "Daily", "Hourly", "Minutes"],
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
                    submit_btn = gr.Button("Plot for the dates")
                    avg_aci_whole = gr.Plot(label="Average ACI over whole timeline")

    with gr.Tab('Plots according to regions'):
        with gr.Column():
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
            acoustic_region_plot = gr.outputs.Image(label="Average ACI over acoustic regions", type="pil")

    with gr.Tab('Clustering'):
        with gr.Column():
            with gr.Row():
                clusters_ideal = gr.Radio(['Choose random number of clusters', 'Get optimum number of clusters'], interactive=True)
                num_clusters = gr.Slider(minimum=1, maximum=10, value=2, step=1,
                                          label="Select the number of clusters", interactive=True, visible=True)
                cluster_indices = gr.CheckboxGroup(['ACI', 'ENT', 'EVN', 'ECV', 'EAS', 'LFC', 'HFC', 'MFC', 'EPS'], label= 'Choose the parameters for clustering')
            clusters_pic = gr.Plot(label="Clusters based on k-means")
            submit_btn_clusters =gr.Button('Plot Clusters', interactive=True)

    def display_options(selected_option):
        if selected_option == 'diel cycle':
            return gr.Text(visible=True)
        else :
            return gr.Text(visible=False)

    submit_btn.click(calculate_plot_whole_year, inputs=[radio_x_axis, radio_groupby, index_select, resolution], outputs=avg_aci_whole)
    submit_btn_2.click(call_plot_aci_values_regions, [plot, hue, region_type], acoustic_region_plot)
    submit_btn_clusters.click(clustering, [clusters_ideal, num_clusters, cluster_indices], clusters_pic)
    radio_x_axis.change(display_options, radio_x_axis, disclaimer)

if __name__ == "__main__":
    demo.launch()
    #demo.launch(server_name='0.0.0.0', server_port=7860)
