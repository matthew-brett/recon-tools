#include "SortingEdges.h"

void Quick_Sort_Edges(Edge * begin, Edge * end, Edge *temp)
{
	float testvalue;
	Edge *temp_head, *temp_tail;
	if((end->Edge_Index ) > (begin->Edge_Index))
	{
		testvalue= begin->Reliability;
		temp_head = begin->next;
		temp_tail = end->previous;
		if(begin->Reliability > end->Reliability)
		{
			Swap_Edges(begin,end,temp);
		}
		
		while(temp_head->Edge_Index <= temp_tail->Edge_Index)
		{
			if(temp_head->Reliability <= testvalue)
			{
				temp_head = temp_head->next;
			}
			else
			{
				Swap_Edges(temp_head,temp_tail,temp);
				temp_tail = temp_tail->previous;
			}
		}
		temp_head = temp_head->previous;
		temp_tail = temp_tail->next;
		Swap_Edges(begin,temp_head,temp);
		Quick_Sort_Edges(begin,temp_head,temp);
		Quick_Sort_Edges(temp_tail,end,temp);
	}
}

void Quick_Sort_Edges1(Edge * begin, Edge * end, Edge *temp,Edge * temp_head, Edge * temp_tail, float testvalue)
{
	if((end->Edge_Index ) > (begin->Edge_Index))
	{
		testvalue= begin->Reliability;
		temp_head = begin->next;
		temp_tail = end->previous;
		if(begin->Reliability > end->Reliability)
		{
			Swap_Edges(begin,end,temp);
		}
		
		while(temp_head->Edge_Index <= temp_tail->Edge_Index)
		{
			if(temp_head->Reliability <= testvalue)
			{
				temp_head = temp_head->next;
			}
			else
			{
				Swap_Edges(temp_head,temp_tail,temp);
				temp_tail = temp_tail->previous;
			}
		}
		temp_head = temp_head->previous;
		temp_tail = temp_tail->next;
		Swap_Edges(begin,temp_head,temp);
		Quick_Sort_Edges(begin,temp_head,temp);
		Quick_Sort_Edges(temp_tail,end,temp);
	}
}

void Swap_Edges(Edge *edge1, Edge *edge2, Edge *temp)
{
	temp->Reliability = edge2->Reliability;
	temp->pointer1 =  edge2->pointer1;
	temp->pointer2 = edge2->pointer2;

	edge2->Reliability = edge1->Reliability;
	edge2->pointer1= edge1->pointer1;
	edge2->pointer2 = edge1->pointer2;

	edge1->Reliability = temp->Reliability;
	edge1->pointer1 = temp->pointer1;
	edge1->pointer2 = temp->pointer2;

}

////////////////////////////////////////Code for Merge Sorting
float cmp(Edge *a, Edge *b)
{
    return a->Reliability - b->Reliability;
}

/*
 * This is the actual sort function. Notice that it returns the new
 * head of the list. (It has to, because the head will not
 * generally be the same element after the sort.) So unlike sorting
 * an array, where you can do
 * 
 *     sort(myarray);
 * 
 * you now have to do
 * 
 *     list = listsort(mylist);
 */
Edge * MergeSort_EdgesS2B(Edge *list, int is_circular, int is_double)
{
    Edge *p, *q, *e, *tail, *oldhead;
    int insize, nmerges, psize, qsize, i;

    /*
     * Silly special case: if `list' was passed in as NULL, return
     * NULL immediately.
     */
    if (!list)
	return NULL;

    insize = 1;

    while (1)
	{
        p = list;
		oldhead = list;		       /* only used for circular linkage */
        list = NULL;
        tail = NULL;

        nmerges = 0;  /* count number of merges we do in this pass */

        while (p) 
		{
            nmerges++;  /* there exists a merge to be done */
            /* step `insize' places along from p */
            q = p;
            psize = 0;
            for (i = 0; i < insize; i++) 
			{
                psize++;
			if (is_circular)
		    q = (q->next == oldhead ? NULL : q->next);
			else
		    q = q->next;
                if (!q) break;
            }

            /* if q hasn't fallen off end, we have two lists to merge */
            qsize = insize;

            /* now we have two lists; merge them */
            while (psize > 0 || (qsize > 0 && q))
			{

                /* decide whether next element of merge comes from p or q */
                if (psize == 0)
				{
		    /* p is empty; e must come from q. */
				e = q; q = q->next; qsize--;
				if (is_circular && q == oldhead) q = NULL;
				} else if (qsize == 0 || !q)
				{
				/* q is empty; e must come from p. */
				e = p; p = p->next; psize--;
				if (is_circular && p == oldhead) p = NULL;
				} else if (cmp(q,p) >= 0 ) //(cmp(p,q) <= 0)   //I Change in this line<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				{
		    /* First element of p is lower (or same);
		     * e must come from p. */
				e = p; p = p->next; psize--;
				if (is_circular && p == oldhead) p = NULL;
				} else
				{
				/* First element of q is lower; e must come from q. */
				e = q; q = q->next; qsize--;
				if (is_circular && q == oldhead) q = NULL;
			}

                /* add the next element to the merged list */
			if (tail) {
		    tail->next = e;
		} else {
		    list = e;
		}
		if (is_double) {
		    /* Maintain reverse pointers in a doubly linked list. */
		    e->previous = tail;
		}
		tail = e;
            }

            /* now p has stepped `insize' places along, and q has too */
            p = q;
        }
	if (is_circular) {
	    tail->next = list;
	    if (is_double)
		list->previous = tail;
	} else
	    tail->next = NULL;

        /* If we have done only one merge, we're finished. */
        if (nmerges <= 1)   /* allow for nmerges==0, the empty list case */
            return list;

        /* Otherwise repeat, merging lists twice the size */
        insize *= 2;
    }
}

Edge * MergeSort_EdgesB2S(Edge *list, int is_circular, int is_double)
{
    Edge *p, *q, *e, *tail, *oldhead;
    int insize, nmerges, psize, qsize, i;

    /*
     * Silly special case: if `list' was passed in as NULL, return
     * NULL immediately.
     */
    if (!list)
	return NULL;

    insize = 1;

    while (1)
	{
        p = list;
		oldhead = list;		       /* only used for circular linkage */
        list = NULL;
        tail = NULL;

        nmerges = 0;  /* count number of merges we do in this pass */

        while (p) 
		{
            nmerges++;  /* there exists a merge to be done */
            /* step `insize' places along from p */
            q = p;
            psize = 0;
            for (i = 0; i < insize; i++) 
			{
                psize++;
			if (is_circular)
		    q = (q->next == oldhead ? NULL : q->next);
			else
		    q = q->next;
                if (!q) break;
            }

            /* if q hasn't fallen off end, we have two lists to merge */
            qsize = insize;

            /* now we have two lists; merge them */
            while (psize > 0 || (qsize > 0 && q))
			{

                /* decide whether next element of merge comes from p or q */
                if (psize == 0)
				{
		    /* p is empty; e must come from q. */
				e = q; q = q->next; qsize--;
				if (is_circular && q == oldhead) q = NULL;
				} else if (qsize == 0 || !q)
				{
				/* q is empty; e must come from p. */
				e = p; p = p->next; psize--;
				if (is_circular && p == oldhead) p = NULL;
				} else if (cmp(q,p) <= 0 ) //(cmp(p,q) <= 0)   //I Change in this line<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
				{
		    /* First element of p is lower (or same);
		     * e must come from p. */
				e = p; p = p->next; psize--;
				if (is_circular && p == oldhead) p = NULL;
				} else
				{
				/* First element of q is lower; e must come from q. */
				e = q; q = q->next; qsize--;
				if (is_circular && q == oldhead) q = NULL;
			}

                /* add the next element to the merged list */
			if (tail) {
		    tail->next = e;
		} else {
		    list = e;
		}
		if (is_double) {
		    /* Maintain reverse pointers in a doubly linked list. */
		    e->previous = tail;
		}
		tail = e;
            }

            /* now p has stepped `insize' places along, and q has too */
            p = q;
        }
	if (is_circular) {
	    tail->next = list;
	    if (is_double)
		list->previous = tail;
	} else
	    tail->next = NULL;

        /* If we have done only one merge, we're finished. */
        if (nmerges <= 1)   /* allow for nmerges==0, the empty list case */
            return list;

        /* Otherwise repeat, merging lists twice the size */
        insize *= 2;
    }
}
