select  m.data, a.pesoorigem, a.idpotreiro
from animais a, medicao m
where a.idmedicao = m.id and a.idpotreiro = 2
order by m.data asc