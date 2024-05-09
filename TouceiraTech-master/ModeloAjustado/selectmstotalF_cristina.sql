select
cast(avg(p.mstotal) as NUMERIC(7,2)) as mstotal
from potreiro_cristina t, pastagem_cristina p, medicao_cristina m 
where m.data is not NULL and p.idmedicao=m.id_medicao and p.dentrofora='F' and p.idpotreiro=9
group by p.dentrofora, m.data
order by m.data;