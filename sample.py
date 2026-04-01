{ Sample Pascal program exercising all supported constructs }

program FizzBuzz;

var
  i     : Integer;
  found : Boolean;
  grade : Char;

begin
  { --- simple assignment and arithmetic --- }
  i := 1;
  found := false;

  { --- while loop --- }
  while i <= 20 do
  begin
    if (i mod 3 = 0) and (i mod 5 = 0) then
      writeln('FizzBuzz')
    else if i mod 3 = 0 then
      writeln('Fizz')
    else if i mod 5 = 0 then
      writeln('Buzz')
    else
      writeln(i);
    i := i + 1
  end;

  { --- for loop --- }
  writeln('Squares 1-5:');
  for i := 1 to 5 do
    writeln(i * i);

  { --- readln and char --- }
  writeln('Enter a grade (A/B/C): ');
  readln(grade);
  if grade = 'A' then
    writeln('Excellent!')
  else if grade = 'B' then
    writeln('Good')
  else
    writeln('Keep trying');

  { --- boolean logic --- }
  found := (i > 3) and not (i = 10);
  if found then
    writeln('Found!')
  else
    writeln('Not found')
end.
