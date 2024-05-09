import psycopg2
import numpy as np
from datetime import datetime
import math
from pyeto import fao56_penman_monteith, deg2rad, net_rad, psy_const, delta_svp, celsius2kelvin, wind_speed_2m, et_rad, sol_dec, sunset_hour_angle, inv_rel_dist_earth_sun, cs_rad, sol_rad_from_t, avp_from_rhmean, svp_from_t, delta_svp, avp_from_rhmean, atm_pressure, net_out_lw_rad, psy_const, net_in_sol_rad


now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução iniciada em: " + str(now) + "\n\n")

try:   
#Conecta com o BD
    con = psycopg2.connect("host='localhost' port='5432' dbname='db_2019' user='postgres' password='12345'")
	#con = psycopg2.connect("host='localhost' port='5432' dbname='pastagemOutlier' user='postgres' password='123456'")  
    cur = con.cursor()
except:
	log.write("\nFalha na conexão com o BD!")

file = open("selectevapo.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)

tdo= cur.fetchall()
d = 1

for x in tdo:
    if (x[1] is None) or (x[2] is None) or (x[3] is None) or (x[4] is None) or (x[5] is None):
        evapo_calc = 0  
    else:
        temp_med = x[2]
        tmax = x[3] + 273.15 #273.15 para converter ºC para K
        temp_max_dia = x[3]
        tmin = x[1] + 273.15
        temp_min_dia = x[1]
        umid_med = x[4]
        med_vel_vento = x[5]
        
        lat = deg2rad(-31.347801)
        dia = sol_dec(d)
        por_do_sol = sunset_hour_angle(lat, dia)
        inv_dia = inv_rel_dist_earth_sun(d)
        #tempMin = celsius2kelvin(temp_min_dia)
        #tempMax = celsius2kelvin(temp_max_dia)

        rad_extraterrestre = et_rad(lat, dia, por_do_sol, inv_dia)

        rad_ceu = cs_rad(lat, rad_extraterrestre)
        sol_rad = sol_rad_from_t(rad_extraterrestre, rad_ceu, temp_min_dia, temp_max_dia, False)

        rad_liq_onda_curta = net_in_sol_rad(sol_rad, albedo = 0.23)

        vapor_dia_min = svp_from_t(temp_min_dia)
        vapor_dia_max = svp_from_t(temp_max_dia)

        pressao_real = avp_from_rhmean(vapor_dia_min, vapor_dia_max, umid_med)

        rad_liq_onda_longa = net_out_lw_rad(tmin, tmax, sol_rad, rad_ceu, pressao_real)

        rad = net_rad(rad_liq_onda_curta, rad_liq_onda_longa)
        
        #####Saturação da pressão do vapor########
        pressao_vapor = svp_from_t(temp_med)

        ######################################
        t =  celsius2kelvin(temp_med)
        ####Delta saturação da pressão do vapor########

        delta = delta_svp(temp_med)

        ######################################

        #####Constante PSY################
        alt = 226
        pressao_atm = atm_pressure(alt)
        psy = psy_const(pressao_atm)
        
        evapo_calc = fao56_penman_monteith(rad, t, med_vel_vento, pressao_vapor, pressao_real, delta, psy, shf=0.0)
        
    
    print(x[0],x[1],x[2],x[3],x[4], x[5], d, evapo_calc)
    
    cur.execute("update clima_evapo set evapo=%s where data= '%s';" % (evapo_calc, x[0].isoformat()))
    if (x[0].isoformat() == "2007-01-01") or (x[0].isoformat() == "2008-01-01") or (x[0].isoformat() == "2009-01-01") or (x[0].isoformat() == "2010-01-01") or (x[0].isoformat() == "2011-01-01") or (x[0].isoformat() == "2012-01-01") or (x[0].isoformat() == "2013-01-01") or (x[0].isoformat() == "2014-01-01") or (x[0].isoformat() == "2015-01-01") or (x[0].isoformat() == "2016-01-01") or (x[0].isoformat() == '2017-01-01') or (x[0].isoformat() == '2018-01-01') or (x[0].isoformat() == '2019-01-01') or (x[0].isoformat() == "2020-01-01") or (x[0].isoformat() == "2021-01-01"):
        d=1
    else:
        d+=1
    
con.commit()
