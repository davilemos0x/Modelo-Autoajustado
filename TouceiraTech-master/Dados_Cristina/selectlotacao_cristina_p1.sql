select
a.data,
a.ca
from animais_cristina a, pastagem_cristina p
where p.idpotreiro=a.potreiro and p.idpotreiro='1'
group by a.ca, a.data
order by a.data