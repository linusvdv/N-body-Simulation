from astroquery.jplhorizons import Horizons
from astropy.constants import au
import astropy.units as u

epoch = 2451545.0
planet_ids = {
    "Sun": 10,
    "Mercury": 199,
    "Venus": 299,
    "Moon": 301,
    "Earth": 399,
    "Mars": 499,
    "Jupiter": 599,
    "Saturn": 699,
    "Uranus": 799,
    "Neptune": 899
}
masses = {
    10: 1.9885e30,     # Sun
    199: 3.3011e23,    # Mercury
    299: 4.8675e24,    # Venus
    399: 5.9722e24,    # Earth
    301: 7.342e22,     # Moon
    499: 6.4171e23,    # Mars
    599: 1.8982e27,    # Jupiter
    699: 5.6834e26,    # Saturn
    799: 8.6810e25,    # Uranus
    899: 1.0241e26,    # Neptune
}



def get_state_vector(body_id):
    vec = Horizons(id=str(body_id), id_type=None,
                   location='@sun', epochs=epoch).vectors()
    # raw numbers in AU and AU/day
    x_au, y_au, z_au     = vec['x'][0], vec['y'][0], vec['z'][0]
    vx_au_d, vy_au_d, vz_au_d = vec['vx'][0], vec['vy'][0], vec['vz'][0]

    # positions: multiply by au, then convert to meters
    pos = [ (coord * au).to(u.m) for coord in (x_au, y_au, z_au) ]

    # velocities: multiply by (au/day), then convert to m/s
    vel = [ (v * au/u.d).to(u.m/u.s) for v in (vx_au_d, vy_au_d, vz_au_d) ]

    return pos, vel, masses[body_id]

for i, (key, id) in enumerate(planet_ids.items()):
    pos, vel, mass = get_state_vector(id)
    print("bodies[{}] = {{{{{:.6e}, {:.6e}, {:.6e}}}, {{{:.6e}, {:.6e}, {:.6e}}}, {{0, 0, 0}}, {}}};".format(
        i,
        pos[0].value, pos[1].value, pos[2].value,
        vel[0].value, vel[1].value, vel[2].value,
        mass))
