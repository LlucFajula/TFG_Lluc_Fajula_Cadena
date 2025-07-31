import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge, nearest_points
from shapely.geometry.base import BaseGeometry


# 1) rangs geogràfics de la Península Ibèrica

lat_min, lat_max = 36.6, 43.5  # Graus nord 
lon_min, lon_max = -9.5, 3.333  # Graus est 

# 2) Parametres temporals
start_date = "2025-04-07"
end_date   = "2025-04-13"
timezone   = "Europe/Madrid"

# 3) Carrega polígon pen iberica i genera grid de punts al llarg de les interseccions

# descarrega el GeoJSON, filtra Espanya+Portugal
GEOJSON_URL = (
    "https://raw.githubusercontent.com/"
    "PublicaMundi/MappingAPI/master/data/geojson/countries.geojson"
)
world = gpd.read_file(GEOJSON_URL)

iberia = world[world["name"].isin(["Spain", "Portugal"])]

iberia_polygon = iberia.geometry.union_all() # unifica 

n_i, n_j = 35, 35  # grid 35x35 

lats = np.linspace(lat_max, lat_min, n_i)

valid_points_list = []  # per guardar (i, j, lat, lon)
#loop d'interseccions
for i, lat in enumerate(lats):
    horizontal_line = LineString([(lon_min, lat), (lon_max, lat)]) #linia horitzontal per latitud
    
    intersection = iberia_polygon.intersection(horizontal_line) #intersecció
    # Resultat de intersecció
    line_segments = []
    if intersection.is_empty:
        print(f"No interseccio a lat {lat:.2f}")
        continue
    elif isinstance(intersection, LineString):
        line_segments.append(intersection)
    elif isinstance(intersection, MultiLineString):
        line_segments.extend(list(intersection.geoms))
    elif isinstance(intersection, BaseGeometry):
         # si hi ha problema de geometria
         print(f"Unexpected geometry type a lat {lat:.2f}: {type(intersection)}")
         continue
    else:
         print(f"Problema inesperat a lat {lat:.2f}: {intersection}")
         continue

    # Combina coordenades de segments de la lat
    all_coords = []
    for segment in line_segments:
        all_coords.extend(list(segment.coords))

    if not all_coords:
        print(f"No coordinates found for line segments at latitude {lat:.2f}")
        continue
    # Calcula llargades de segments
    cumulative_lengths = [0.0]
    for k in range(1, len(all_coords)):
        p1 = Point(all_coords[k-1])
        p2 = Point(all_coords[k])
        cumulative_lengths.append(cumulative_lengths[-1] + p1.distance(p2))

    total_length = cumulative_lengths[-1]
#si no hi ha llargada 
    if total_length == 0:
        print(f"No hi ha segment a {lat:.2f}")
        continue

    # Distribueix n_j punts en segment
    desired_lengths = [(j + 1) / (n_j + 1) * total_length for j in range(n_j)]

    # coordenades de les llargades de subsegment
    interpolated_points = []
    for desired_len in desired_lengths:
        # posicions dins el segment
        k = next(k for k, cum_len in enumerate(cumulative_lengths) if cum_len >= desired_len)

        # per si dona error, no hauria de passar
        if k == 0:
             point_coords = all_coords[0]
        else:
             start_point = Point(all_coords[k-1])
             end_point = Point(all_coords[k])
             segment_length = cumulative_lengths[k] - cumulative_lengths[k-1]
             if segment_length == 0: 
                  point_coords = all_coords[k]
             else:
                  fraction_in_segment = (desired_len - cumulative_lengths[k-1]) / segment_length
                  # interpola
                  interp_x = start_point.x + fraction_in_segment * (end_point.x - start_point.x)
                  interp_y = start_point.y + fraction_in_segment * (end_point.y - start_point.y)
                  point_coords = (interp_x, interp_y)


        interpolated_points.append(Point(point_coords))

    # afegeix punts a la llista 
    for j, point in enumerate(interpolated_points):
        valid_points_list.append((i, j, point.y, point.x))


M = len(valid_points_list)
print(f"Punts Generats: {M}")
if M != n_i * n_j:
    print(f"Error, {M} punts, menys dels esperats")



# 4) Inicialitzar l'array numpy per timestamps
url = "https://archive-api.open-meteo.com/v1/archive"
# fem servir el primer punt valid per extreure timestamps
if not valid_points_list:
    print("No punts valids")
    timestamps = None
    num_times = 0
else:
    _, _, lat0, lon0 = valid_points_list[0]
    params = {
        "latitude": lat0,
        "longitude": lon0,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": timezone
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data0 = resp.json()
        timestamps = pd.to_datetime(data0["hourly"]["time"])
        num_times = len(timestamps)
    except requests.exceptions.RequestException as e:
        print(f"Error en la petició inicial: {e}")
        timestamps = None
        num_times = 0


# 5) Continuem timestamps correctes i hi ha punts valids

if timestamps is not None and num_times > 0 and valid_points_list:
    total_valid_points = len(valid_points_list)
    arr = np.empty((num_times, total_valid_points), dtype=float)
    arr[:] = np.nan
    # itera sobre cada punt vàlid
    for idx, (i, j, lat, lon) in enumerate(valid_points_list):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": timezone
        }
        max_retries = 3
        for attempt in range(1, max_retries+1):
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                temps = r.json()["hourly"]["temperature_2m"]
                arr[:, idx] = temps
                print(f"[OK] ({i},{j}) → idx={idx}/{total_valid_points-1}")
                break
            except Exception as e:
                print(f"[{attempt}/{max_retries}] Punt ({i},{j}) error: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"[FAIL] Punt ({i},{j}) inhabilitat, queda com NaN")

    time.sleep(0.2) #no sobrecarregar api, dona error

    
    # 6) Construcció del dataframe d'una sola passada
    cols = [f"T_{i}_{j}" for (i,j,_,_) in valid_points_list]
    df_grid = pd.DataFrame(arr, index=timestamps, columns=cols)

    point_coords_data = [{'Point_ID': f'T_{i}_{j}', 'Latitude': lat, 'Longitude': lon} for (i, j, lat, lon) in valid_points_list]
    df_point_coords = pd.DataFrame(point_coords_data)


    
    # 7) Exportació a Excel amb dues pestanyes
    output_excel = "temperatures_grid_35x35_iberia.xlsx"
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df_grid.to_excel(writer, sheet_name="Grid_Temps")
        df_point_coords.to_excel(writer, sheet_name="Coordenades", index=False)
    print(f"Dades exportades a: {output_excel}")
    
    # 8) Visualització 
    plt.figure(figsize=(8,8))
    # contorn 
    gpd.GeoSeries(iberia_polygon).boundary.plot(color="black", linewidth=1)
    # punts
    xs = [lon for (_,_,_,lon) in valid_points_list]
    ys = [lat for (_,_,lat,_) in valid_points_list]
    plt.scatter(xs, ys, c="red", s=10)
    plt.xlabel("Longitud"); plt.ylabel("Latitud")
    plt.title(f"{total_valid_points} punts distribuïts dins Iberia")
    plt.tight_layout()
    plt.show()


else:
    print("Error: No s'han obtingut timestamps o no s'han trobat punts valids")