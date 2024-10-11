todo.txt

## To Do

to access the shell
docker compose exec -it flask_app /bin/bash


for mongodb
docker compose exec -it mongodb /bin/bash
mongosh mongodb://admin:secret@localhost:27017/admin



PRIORITIES
6. Add cross-referencing of named entities

STATUS: linking seems to work. unique terms no longer works. need to check or revert or soemthing.



named entity, reprocessing, and fuzzy matching

1. reprocess and add NER to documents, priomed for fuzzy matching
2. clik to mfuzzy match and open new document


            spacey? spacey vs AI
            how to reconcile different named entities?

            first entry or something?

            toponyms

            forced to see something not looking for

            fuzzy matching

            click on that and open a new tab and show all the files


            head injury, missing pictures

            remove blur

            add contrast


            llama 3.1

            move documents name to .env file







##Feature to do list
0. Shift other PC to Docker

1. Convert setup process to a part of settings or a new page. It should be able to add to the DB
2. Create login splash page

4. Address weirdness of base file and index.html. It is unseemly 


7. Restore adding export from list
8. Add "Select all" for export
9. Clean file description to remove extension in title
10. Create a way to do a search and then add that result to the DB
11. Backup and restore database

13. Color code sections of JSON expansion
14. Need to implement a sort for the file results, so that this is in order: 	File	Summary
	RDApp-630550Fox053.jpg.json	The document contains a handwritten signature of an individual named M. Johnson, along with the year 1919.
	RDApp-630550Fox072.jpg.json	This document is a surgeon's first report of an accident for the Baltimore & Ohio Railroad-Relief Department. It details an incident involving an individual named E.S. Fry, a laborer, who resides in All Around O. The report notes injuries sustained: a contusion and a cut lip, with the mention of a broken face due to a tool. The probable duration of disablement is stated to be short. Additionally, it provides a brief account of how the accident occurred, indicating involvement with a train.
	RDApp-630550Fox059.jpg.json	The document appears to be a handwritten note addressed to Mr. Martin from someone requesting approval from Dr. Smith, a company surgeon, regarding a matter likely related to medical or health concerns.
	RDApp-630550Fox014.jpg.json	This document is a correspondence from the Office of General Claim Agent of The Baltimore and Ohio Railroad Company, dated December 8, 1920. It refers to a bill from The Peoples Hospital for services rendered to E. L. Fox, a train rider who was injured at Cuyahoga Falls, Ohio, on October 31, 1920. The bill is being sent to Mr. W. J. Dudley, Superintendent of the Relief Department, for voucher processing. The document indicates that Mr. Fox was a member of the Relief Department at the time of his injury. Additionally, there is a note referencing a letter related to this bill dated 16th of December, 1920.
	RDApp-630550Fox062.jpg.json	This document is a telegram from the Baltimore and Ohio Railroad Company to the Superintendent of City Hospital in Akron, Ohio, dated August 5th, 1921. It refers to a bill concerning an individual named E. L. Fox and requests further communication regarding the matter.


##Notes on how to
1. Inside the util folder is delete_db.py which needs to be run from inside a container in order to delete the database. You will need to delete the database if you change the setup or structure.
   a. docker exec -it flask_app /bin/bash
   b. util/delete_db.py

