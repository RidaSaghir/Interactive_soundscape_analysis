import pandas as pd
import plotly.express as px


def cluster_occurrence(data_clustered, which_cluster, cluster_x_axis, cluster_hue):
    df = data_clustered.copy()  # Make a copy to avoid modifying the original DataFrame
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert 'Timestamp' column to datetime type
    df.set_index('Timestamp', inplace=True)  # Set 'Timestamp' as the index
    df = df[['Year', 'Month', 'Week', 'Day', 'Hour', 'Minute', 'KMeans_Cluster']]
    cluster_df = df[df['KMeans_Cluster'] == int(which_cluster)]
    print(cluster_df)
    if cluster_x_axis == 'Year cycle':
        if cluster_hue == 'Year':
            df_filtered = cluster_df[['Year', 'Month']]
            df_count = df_filtered.groupby(['Year', 'Month']).size().reset_index(name='Count')
            fig = px.bar(df_count, x='Month', y='Count', color='Year',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')
        if cluster_hue == 'Month':
            df_filtered = cluster_df[['Year', 'Month']]
            df_count = df_filtered.groupby(['Year', 'Month']).size().reset_index(name='Count')

            fig = px.bar(df_count, x='Month', y='Count', color='Month',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        if cluster_hue == 'Day':
            df_filtered = cluster_df[['Day', 'Month']]
            df_count = df_filtered.groupby(['Day', 'Month']).size().reset_index(name='Count')

            fig = px.bar(df_count, x='Month', y='Count', color='Day',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        if cluster_hue == 'Hour':
            df_filtered = cluster_df[['Hour', 'Month']]
            df_count = df_filtered.groupby(['Hour', 'Month']).size().reset_index(name='Count')

            fig = px.bar(df_count, x='Month', y='Count', color='Hour',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        fig.update_layout(title='Cluster {} occurrences'.format(which_cluster), xaxis_title='Months',
                      yaxis_title='Cluster {} occurences'.format(which_cluster))


    if cluster_x_axis == 'Diel cycle':
        if cluster_hue == 'Year':
            df_filtered = cluster_df[['Year', 'Hour']]
            df_count = df_filtered.groupby(['Year', 'Hour']).size().reset_index(name='Count')

            fig = px.bar(df_count, x='Hour', y='Count', color='Year',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')
        if cluster_hue == 'Month':
            df_filtered = cluster_df[['Month', 'Hour']]
            df_count = df_filtered.groupby(['Month', 'Hour']).size().reset_index(name='Count')

            fig = px.bar(df_count, x='Hour', y='Count', color='Month',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        if cluster_hue == 'Day':
            df_filtered = cluster_df[['Day', 'Hour']]
            df_count = df_filtered.groupby(['Day', 'Hour']).size().reset_index(name='Count')

            fig = px.bar(df_count, x='Hour', y='Count', color='Day',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        if cluster_hue == 'Hour':
            df_filtered = cluster_df[['Hour']]
            df_count = df_filtered.groupby(['Hour']).size().reset_index(name='Count')

            fig = px.bar(df_count, x='Hour', y='Count', color='Hour',
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        fig.update_layout(title='Cluster {} occurrences'.format(which_cluster), xaxis_title='Months',
                      yaxis_title='Cluster {} occurences'.format(which_cluster))

    if cluster_x_axis == 'Linear cycle':
        if cluster_hue == 'Year':
            df_count = cluster_df.groupby(cluster_df.index.date).size().reset_index(name='Count')
            df_count['index'] = pd.to_datetime(df_count['index'])
            df_count.set_index('index', inplace=True)
            fig = px.bar(df_count, x=df_count.index.date, y='Count', color=df_count.index.year,
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')
        if cluster_hue == 'Month':
            df_count = cluster_df.groupby(cluster_df.index.date).size().reset_index(name='Count')
            df_count['index'] = pd.to_datetime(df_count['index'])
            df_count.set_index('index', inplace=True)
            fig = px.bar(df_count, x=df_count.index.date, y='Count', color=df_count.index.month,
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        if cluster_hue == 'Day':
            df_count = cluster_df.groupby(cluster_df.index.date).size().reset_index(name='Count')
            df_count['index'] = pd.to_datetime(df_count['index'])
            df_count.set_index('index', inplace=True)
            fig = px.bar(df_count, x=df_count.index.date, y='Count', color=df_count.index.day,
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        if cluster_hue == 'Hour':
            df_count = cluster_df.groupby([cluster_df.index.date, cluster_df.index.hour]).size().reset_index(
                name='Count')
            df_count['index'] = pd.to_datetime(df_count['index'])
            df_count.set_index('index', inplace=True)
            fig = px.bar(df_count, x=df_count.index.date, y='Count', color=df_count.index.hour,
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

        fig.update_layout(title='Cluster {} occurrences'.format(which_cluster), xaxis_title='Months',
                              yaxis_title='Cluster {} occurences'.format(which_cluster))
    return fig