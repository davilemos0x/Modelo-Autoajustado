--Inf
/*
select
p.dentrofora, 
m.data,
cast((sum(p.media*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as media, 
cast((sum(p.mstotal*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as mstotal, 
cast((sum((p.msanoni/(p.msanoni+p.msoutras))*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as pcanonni,
EXTRACT(MONTH FROM m.data) as "Mes",
EXTRACT(YEAR FROM m.data) as "Ano"
from subarea s, potreiro t, pastagem p, medicao m 
where m.data is not null and s.idpotreiro=t.id and p.idsubarea=s.id and p.idmedicao=m.id and p.mstotal!=0 and t.id=4
group by p.dentrofora, m.data
order by m.data;
*/
--Mira
select
p.dentrofora, 
m.data,
cast((sum(p.media*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as media, 
cast((sum(p.mstotal*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as mstotal, 
cast((sum((p.msanoni/(p.msanoni+p.msoutras))*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as pcanonni,
EXTRACT(MONTH FROM m.data) as "Mes",
EXTRACT(YEAR FROM m.data) as "Ano"
from subarea s, potreiro t, pastagem p, medicao m 
where m.data between '2013-12-20' and '2018-02-19' and s.idpotreiro=t.id and p.idsubarea=s.id and p.idmedicao=m.id and p.mstotal!=0 and t.id=3
group by p.dentrofora, m.data
order by m.data;
