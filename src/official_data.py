from astroquery.jplhorizons import Horizons
from astropy.constants import au
import astropy.units as u

epoch = 2451545.0

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

    return pos, vel

# Earth
earth_pos, earth_vel = get_state_vector(399)
# Sun
sun_pos, sun_vel     = get_state_vector(10)
# Moon
moon_pos, moon_vel   = get_state_vector(301)


# Earth
print("Earth position (m):", earth_pos[0].value, earth_pos[1].value, earth_pos[2].value)
print("Earth velocity (m/s):", earth_vel[0].value, earth_vel[1].value, earth_vel[2].value)
print("Earth: {{{:.6e}, {:.6e}, {:.6e}}}, {{{:.6e}, {:.6e}, {:.6e}}}".format(
    earth_pos[0].value, earth_pos[1].value, earth_pos[2].value,
    earth_vel[0].value, earth_vel[1].value, earth_vel[2].value))

# Sun
print("Sun position (m):", sun_pos[0].value, sun_pos[1].value, sun_pos[2].value)
print("Sun velocity (m/s):", sun_vel[0].value, sun_vel[1].value, sun_vel[2].value)
print("Sun:   {{{:.6e}, {:.6e}, {:.6e}}}, {{{:.6e}, {:.6e}, {:.6e}}}".format(
    sun_pos[0].value, sun_pos[1].value, sun_pos[2].value,
    sun_vel[0].value, sun_vel[1].value, sun_vel[2].value))

# Moon
print("Moon position (m):", moon_pos[0].value, moon_pos[1].value, moon_pos[2].value)
print("Moon velocity (m/s):", moon_vel[0].value, moon_vel[1].value, moon_vel[2].value)
print("Moon:  {{{:.6e}, {:.6e}, {:.6e}}}, {{{:.6e}, {:.6e}, {:.6e}}}".format(
    moon_pos[0].value, moon_pos[1].value, moon_pos[2].value,
    moon_vel[0].value, moon_vel[1].value, moon_vel[2].value))
