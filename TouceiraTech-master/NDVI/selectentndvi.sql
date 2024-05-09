select 
p.data, 
cast((sum(p.ndvi_medio*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as media,
EXTRACT(MONTH FROM p.data) as "Mes",
EXTRACT(YEAR FROM p.data) as "Ano"  
from subarea s, potreiro t, ndvi p, medicao m 
where m.data is not NULL and s.idpotreiro=t.id and p.idsubarea=s.id and t.id=1
group by p.data 
order by p.data;
