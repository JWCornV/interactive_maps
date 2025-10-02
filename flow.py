import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from ortools.graph.python import min_cost_flow

def flow_redistrict(gdf, num_districts, max_iter=100):
    """
    Partition VTDs into balanced districts using a min-cost-flow clustering approach.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain:
          - 'centroid': shapely Point geometry representing the centroid of each VTD
          - 'total': population count for each VTD (NaN or 0 values are replaced with 1
                     so that every VTD is assigned to a district)
    num_districts : int
        Number of districts to create.
    max_iter : int, default 100
        Maximum number of iterations in the iterative centroid update loop.

    Notes
    -----
    - The algorithm works like balanced K-means:
        1. Randomly initialize district centers from VTD centroids.
        2. Use min-cost-flow to assign VTD populations to districts so each district
           has nearly equal population and assignments minimize distance.
        3. If a VTD is split across multiple districts, assign it to the district
           that receives the majority of its population flow. (This is where there is
           some improvement to be made).
        4. Update district centers as the mean of centroids assigned to each district.
        5. Repeat until district centers converge or `max_iter` is reached.

    Returns None (The input GeoDataFrame `gdf` is modified in-place.)
    """
    
    # Assign dummy population of 1 for unpopulated VTDs just so they get assigned to a district
    adjusted_population = gdf['total'].fillna(1).replace(0, 1)
    
    vtd_points = np.array([(point.x, point.y) for point in gdf['centroid']])
    num_vtds = len(gdf)

    if 'district' in gdf.columns:
        assert(num_districts == gdf['district'].nunique())
        print('Reusing centers')
        centroids = gdf.geometry.centroid
        coords = np.column_stack((centroids.x, centroids.y))
        
        # Take the mean of centroid coordinates within each district
        district_centers = np.vstack(
            gdf.groupby('district')
               .apply(lambda df: coords[df.index].mean(axis=0))
               .to_numpy()
        )
    else:
        district_centers = vtd_points[np.random.choice(vtd_points.shape[0], num_districts, replace=False)]
    
    for iteration in range(max_iter):
        print('Iteration:', iteration)

        # Raw Euclidean distances have some issues with concave geometries, e.g. coastal bays
        distances = cdist(vtd_points, district_centers)
        
        # Create solver
        mcf = min_cost_flow.SimpleMinCostFlow()
    
        # District sizes need to be exact for flow algorithm so track remainder
        total_pop = adjusted_population.sum()
        base = total_pop // num_districts
        remainder = total_pop % num_districts
        district_sizes = [base + 1 if i < remainder else base for i in range(num_districts)]
    
        # Sources are VTDs, index starts at 0, sinks are districts, start from num_vtds
        for i, cap in enumerate(adjusted_population):
            for j in range(num_districts):
                cost = distances[i][j]
                mcf.add_arc_with_capacity_and_unit_cost(i, j + num_vtds, cap, int(cost))
        
        # Each VTD has flow equal to its population
        for i, cap in enumerate(adjusted_population):
            mcf.set_node_supply(i, cap)

        # Each district can accept flow equal to its desired population
        for j, dist_size in enumerate(district_sizes):
            mcf.set_node_supply(j + num_vtds, -dist_size)

        # Solve
        status = mcf.solve()
        assert(status == mcf.OPTIMAL)
        df_results = pd.DataFrame([{'vtd': mcf.tail(i), 'dst': mcf.head(i) - num_vtds, 'flow': mcf.flow(i), 'cost': mcf.unit_cost(i)}
                                  for i in range(mcf.num_arcs()) if mcf.flow(i) > 0])

        # Assign VTDs with flow to multiple districts to majority district
        # This could probably be improved to balance final population better although
        # often the minority-flow district is on the other side of the state for some reason
        df_districts = df_results.loc[df_results.groupby('vtd')['flow'].idxmax().values]
        df_districts = df_districts.sort_values(by='vtd').reset_index(drop=True)
        
        # Update centers to new district centroids
        new_centers = np.array([vtd_points[df_districts[df_districts['dst'] == j]['vtd'].values].mean(axis=0) 
                                for j in range(num_districts)])
        # Check convergence
        if np.allclose(new_centers, district_centers, rtol=1e-6, atol=1e-8):
            print('Centers converged')
            gdf['district'] = (df_districts['dst'] + 1).values
            break
        if iteration == max_iter - 1:
            print('Max Iter')
            gdf['district'] = (df_districts['dst'] + 1).values
            break
        
        district_centers = new_centers
