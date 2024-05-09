select  m.data, a.pesoorigem
from animais a, medicao m
where a.idmedicao = m.id and a.idpotreiro = 1
order by m.data asc