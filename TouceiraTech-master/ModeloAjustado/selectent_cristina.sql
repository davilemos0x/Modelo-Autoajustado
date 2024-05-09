select
p.dentrofora, 
m.data, 
cast(avg(p.media) as NUMERIC(7,2)) as media, 
cast(avg(p.mstotal) as NUMERIC(7,2)) as mstotal,
EXTRACT(MONTH FROM m.data) AS "Mes",
EXTRACT(YEAR FROM m.data) AS "Ano",
p.idpotreiro,
cast(avg(p.id) as numeric(7,0)) as "id"
from potreiro_cristina t, pastagem_cristina p, medicao_cristina m 
where m.data is not null and p.idmedicao=m.id_medicao and p.idpotreiro=9
group by p.dentrofora, m.data, idpotreiro
order by id;