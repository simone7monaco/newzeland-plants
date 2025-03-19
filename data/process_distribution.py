from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import multiprocessing

distributions_dir = Path('Ferns/Distribution layers')



def load_raster(filename, downsample=5):
    ar = xr.open_dataarray(filename, engine='rasterio')
    ar = ar.coarsen(x=downsample, y=downsample, boundary="trim").sum().fillna(0)
    ar['x'] = ar.x.round(1)
    ar['y'] = ar.y.round(1)
    # sum all points having the same coordinates
    ar = ar.groupby(['x', 'y']).sum()
    return ar

def process_portion(files_list):
    for i, raster in enumerate(files_list):
        raster_sum = raster if i == 0 else xr.concat([raster_sum, raster], dim='species').sum('species')

        # raster_sum = raster_sum.interp(x=np.arange(raster_sum.x.min(), raster_sum.x.max(), 0.1),
        #                             y=np.arange(raster_sum.y.min(), raster_sum.y.max(), 0.1))
    return raster_sum

# all_files_list = list(distributions_dir.glob('*.tif'))
batch_size = 5

# raster_chunks = [load_raster(f) for f in distributions_dir.glob('*.tif')]
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    raster_chunks = list(pool.imap(load_raster, distributions_dir.glob('*.tif')))
print(f"Loaded {len(raster_chunks)} rasters")
while len(raster_chunks) != 1:
    # Parallel processing using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        raster_chunks = list(tqdm(pool.imap(process_portion, 
                                            [raster_chunks[i:i+batch_size] for i in range(0, len(raster_chunks), batch_size)]),
                                total=len(raster_chunks)//batch_size))

    # Merge all raster chunks into a final sum
    print(f"Merged {len(raster_chunks)} chunks")

raster_sum = raster_chunks[0]

print(raster_sum.shape)

cluste_samples = raster_sum.isel(band=0).to_pandas().reset_index().melt(id_vars=['x']).rename(columns={'value': 'occurrence'})
cluste_samples = cluste_samples[cluste_samples.occurrence > 0]

# # silouhette score
# n_clusters = range(2, 100)
# silhouette = []
# for n in n_clusters:
#     kmeans = KMeans(n_clusters=n, random_state=0).fit(cluste_samples[['x', 'y']])
#     silhouette.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=50, random_state=0).fit(cluste_samples[['x', 'y']])
cluste_samples['cluster'] = kmeans.labels_
cluste_samples.to_csv(distributions_dir.parent / 'distriubtion_clusters.csv', index=False)
raster_sum.to_netcdf(distributions_dir.parent / 'distriubtion_sum.nc')
